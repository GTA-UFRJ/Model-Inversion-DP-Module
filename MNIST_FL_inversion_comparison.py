import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import argparse
import urllib.request

# --- Valores Padrão ---
START_SEED = 0
NUMBER_OF_SEEDS = 1
NOISE_SCALES = [0.0]
CLASSES = [0]

parser = argparse.ArgumentParser(description="Executa o experimento de inversão de modelo com aprendizado federado.")

parser.add_argument('--start_seed', type=int, default=START_SEED,
                    help=f'A seed inicial para a geração de números aleatórios (padrão: {START_SEED}).')

parser.add_argument('--number_of_seeds', type=int, default=NUMBER_OF_SEEDS,
                    help=f'O número de seeds bem-sucedidas a serem testadas (padrão: {NUMBER_OF_SEEDS}).')

parser.add_argument('--noise_scales', type=str, default=",".join(map(str, NOISE_SCALES)),
                    help=f'Uma string de floats separados por vírgula para as escalas de ruído (padrão: "{",".join(map(str, NOISE_SCALES))}").')
                    
parser.add_argument('--classes', type=str, default=",".join(map(str, CLASSES)),
                    help=f'Uma string de inteiros separados por vírgula para as classes a serem testadas (padrão: "{",".join(map(str, CLASSES))}").')

args = parser.parse_args()

# Atribui os argumentos às variáveis globais
START_SEED = args.start_seed
NUMBER_OF_SEEDS = args.number_of_seeds
try:
    NOISE_SCALES = [float(x.strip()) for x in args.noise_scales.split(',')]
    CLASSES = [int(x.strip()) for x in args.classes.split(',')]
except ValueError as e:
    print(f"Erro ao converter noise_scales ou classes: {e}")
    exit(1)

print("Configurações do Experimento:")
print(f"  START_SEED: {START_SEED}")
print(f"  NUMBER_OF_SEEDS: {NUMBER_OF_SEEDS}")
print(f"  NOISE_SCALES: {NOISE_SCALES}")
print(f"  CLASSES: {CLASSES}")

# Lista para armazenar resultados intermediários (não usada diretamente aqui)
results_data = []

# Carrega um DataFrame existente com resultados ou cria um novo com as colunas apropriadas
results_path = "new_results_with_seed.csv"
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
else:
    results_df = pd.DataFrame(columns=[
        'Label', 'Round', 'Method', 'Noise Scale',
        'Accuracy', 'Loss', 'MSE', 'PSNR', 'SSIM', 'Seed'
    ])

# Funções de ativação e suas derivadas
def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

# Codifica os rótulos em one-hot encoding
def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    return one_hot_Y.T

# Propagação direta (forward) de uma MLP de 1 camada escondida
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Retropropagação (cálculo dos gradientes)
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    # Passa explicitamente o número de classes (10 para o MNIST)
    one_hot_Y = one_hot(Y, 10)

    dZ2 = A2 - one_hot_Y

    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Atualiza os parâmetros com gradiente descendente
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Divide o dataset entre os clientes
def split_data(X, Y, num_clients, samples_per_client=50):
    clients = []
    for _ in range(num_clients):
        indices = np.random.choice(X.shape[1], samples_per_client, replace=False)
        X_client = X[:, indices]
        Y_client = Y[indices]
        clients.append((X_client, Y_client))
    return clients

# Inicializa pesos e bias com valores pequenos aleatórios
def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# Treinamento local em um cliente
def local_train(X, Y, W1, b1, W2, b2, iterations, alpha):
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2

# Agrega os modelos dos clientes (média dos pesos)
def aggregate(models):
    num_clients = len(models)
    aggregated = [np.mean([model[param] for model in models], axis=0) for param in range(len(models[0]))]
    return tuple(aggregated)

# Avalia acurácia do modelo em dados
def evaluate_model(X, Y, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    accuracy = np.mean(predictions == Y) * 100
    return accuracy

# Não usado - Implementado na main
def federated_learning(clients_data, global_model, num_rounds, local_epochs, alpha):
    W1, b1, W2, b2 = global_model
    local_models = []

    for round_num in range(num_rounds):
        #print(f"\n=== Round {round_num + 1}/{num_of_training_rounds} ===")
        client_models = []

        for client_id, (X_client, Y_client) in enumerate(clients_data):
            client_W1, client_b1, client_W2, client_b2 = local_train(X_client, Y_client, W1, b1, W2, b2, local_epochs, alpha)
            client_models.append([client_W1, client_b1, client_W2, client_b2])

        W1, b1, W2, b2 = aggregate(client_models)
        #print("\nGlobal model updated after aggregation.")
        local_models = client_models
    
    return W1, b1, W2, b2, local_models

# Computa MSE, PSNR e SSIM entre imagem original e reconstruída
def compute_metrics(real_image, reconstructed_image):
    mse = np.mean((real_image - reconstructed_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    ssim_value = ssim(real_image, reconstructed_image, data_range=255)
    return mse, psnr, ssim_value

# Gera uma perturbação localizada para imagem
def localized_perturbation(X_sample, perturbation_size=50, sigma=0.05):
    perturbation = np.zeros_like(X_sample)
    active_pixels = np.random.choice(784, perturbation_size, replace=False)
    perturbation[active_pixels] = np.random.normal(0, sigma, size=(perturbation_size, 1))
    return perturbation

# Salva as imagens reconstruídas por rodada e atualiza o CSV de resultados
def display_images(images, losses, matching_images, labels, iterations, round_num,
                   global_model_accuracy, method, results_df, noise_scale,
                   mse=None, psnr=None, ssim_val=None, seed=None):
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    
    # Display the last image without title
    for i, img in enumerate(images):
        if i == len(images)-1:
            axes.imshow(img, cmap='gray')
            axes.axis('off')
    
    # Save the image
    filename = f"{labels[-1]}/{noise_scale} noise/{method} Round {round_num}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    new_row = {
        'Label': labels[-1],
        'Round': round_num,
        'Method': method,
        'Noise Scale': noise_scale,
        'Accuracy': global_model_accuracy,
        'Loss': losses[-1],
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_val,
        'Seed': seed
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df

# Mostra imagens reais da classe-alvo presentes no cliente
def display_matching_images(images, losses, matching_images, labels, iterations):
    fig, axes = plt.subplots(1,len(matching_images), figsize=(15, 5))
    if len(matching_images) > 1:
        for i, img in enumerate(matching_images):
            axes[i].imshow(img.reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Real {labels[0]}")
            axes[i].axis('off')
    elif len(matching_images) == 1:
        axes.imshow(matching_images[0].reshape(28, 28), cmap='gray')
        axes.set_title(f"Real {labels[0]}")
        axes.axis('off')
    
    filename = f"{labels[0]}/Matching Images.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Reconstrução ingênua
def invert_model_naive(W1, b1, W2, b2, target_label, client_data,target_certainty=0.999, max_iterations=3000, milestone=500):
    X_sample = np.zeros((784, 1))

    num_active_pixels = 50
    active_pixels = np.random.choice(784, num_active_pixels, replace=False)
    X_sample[active_pixels] = np.random.uniform(0.5, 1.0, size=(num_active_pixels, 1))
    
    iterations = 0
    certainty = 0

    milestone_images = []
    milestone_certainties = []
    milestone_iterations = []

    X_client, Y_client = client_data

    while certainty < target_certainty and iterations < max_iterations:
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_sample)

        certainty = A2[target_label, 0]

        if iterations % milestone == 0:
            milestone_images.append(X_sample.reshape(28, 28))
            milestone_certainties.append(certainty)
            milestone_iterations.append(iterations)

        perturbation = localized_perturbation(X_sample, perturbation_size=30, sigma=0.02)
        X_sample_new = X_sample + perturbation
        X_sample_new = np.clip(X_sample_new, 0, 1)

        _, _, _, A2_new = forward_prop(W1, b1, W2, b2, X_sample_new)
        certainty_new = A2_new[target_label, 0]

        if certainty_new >= certainty:
            X_sample = X_sample_new
            certainty = certainty_new

        iterations += 1

    milestone_images.append(X_sample.reshape(28, 28))
    milestone_certainties.append(certainty)
    milestone_iterations.append(iterations)

    matching_indices = np.where(Y_client == target_label)[0]
    matching_images = X_client[:, matching_indices].T

    return X_sample, (milestone_images, milestone_certainties, matching_images[:10], [target_label] * len(milestone_images), milestone_iterations)

# Reconstrução por gradientes
def reconstruct_via_loss(W1, b1, W2, b2, target_label, client_data, learning_rate=0.01, iterations=3000, momentum=0.9):
    X_client, Y_client = client_data

    X_reconstructed = np.zeros((784, 1))
    
    velocity = np.zeros_like(X_reconstructed)

    milestone_images = []
    milestone_losses = []
    milestone_iterations = []

    for i in range(1, iterations+1):
        Z1 = np.dot(W1, X_reconstructed) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

        target_one_hot = np.zeros((10, 1))
        target_one_hot[target_label] = 1

        loss = -np.sum(target_one_hot * np.log(A2 + 1e-8))

        dZ2 = A2 - target_one_hot
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dX_reconstructed = np.dot(W1.T, dZ1)

        velocity = -learning_rate * dX_reconstructed
        X_reconstructed += velocity

        X_reconstructed = np.clip(X_reconstructed, 0, 1)

        if i % 500 == 0 or i == iterations - 1:
            milestone_images.append(X_reconstructed.reshape(28, 28))
            milestone_losses.append(loss)
            milestone_iterations.append(i)

    matching_indices = np.where(Y_client == target_label)[0]
    matching_images = X_client[:, matching_indices].T

    return X_reconstructed, (milestone_images, milestone_losses, matching_images[:10], [target_label] * len(milestone_images), milestone_iterations)

# Calcula o mapa de saliência (gradiente da entrada em relação à saída predita)
def compute_saliency_map(W1, b1, W2, b2, X_input):
    """
    Compute and display the saliency map for a given input image.
    """
    # Forward pass
    Z1 = W1 @ X_input + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    # Predicted class
    predicted_label = np.argmax(A2)

    # Compute gradients (saliency map)
    dZ2 = np.zeros_like(A2)
    dZ2[predicted_label] = 1  # Derivative wrt. predicted class
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * deriv_ReLU(Z1)
    saliency_map = np.abs(W1.T @ dZ1)  # Importance of each input pixel

    # Normalize for better visualization
    saliency_map = saliency_map.reshape(28, 28)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Display saliency map
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(X_input.reshape(28, 28), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Saliency Map")
    plt.axis('off')

    plt.show()

# Aplica ruído gaussiano aos pesos para privacidade diferencial local (LDP)
def apply_local_differential_privacy(weights, noise_scale):
    """Apply LDP with additive noise."""
    noisy_weights = [add_noise(w, noise_scale) for w in weights]
    return noisy_weights

# Adiciona ruído gaussiano aos pesos
def add_noise(weight, noise_scale):
    noise = np.random.normal(0, noise_scale, size=weight.shape)
    return weight + noise

def load_mnist_csv(path="mnist_train.csv"):
    url = "https://python-course.eu/data/mnist/mnist_train.csv"

    # Step 1: Check if file exists, otherwise download
    if not os.path.exists(path):
        print(f"{path} not found, downloading from {url}...")
        urllib.request.urlretrieve(url, path)

    # Step 2: Peek at the first line to check for header
    with open(path, "r") as f:
        first_line = f.readline().strip()

    if not first_line.startswith("label,"):
        print("Header missing — adding header...")
        header = "label," + ",".join(
            f"{row}x{col}"
            for row in range(1, 29)
            for col in range(1, 29)
        )
        # Prepend header to file
        with open(path, "r") as f:
            content = f.read()
        with open(path, "w") as f:
            f.write(header + "\n" + content)

    # Step 3: Load into NumPy
    data = pd.read_csv(path).to_numpy()
    return data

# Example usage
data = load_mnist_csv()
exception_counter = 0

# Início
if __name__ == "__main__":
    
    # --- ESTRATÉGIA: Parar quando a ÚLTIMA classe atingir o número de seeds desejado ---
    # Rastreia as seeds bem-sucedidas para cada classe individualmente.
    successful_seeds_per_class = {c: set() for c in CLASSES}
    
    current_seed = START_SEED
    
    # A classe de referência para parar o experimento é a última da lista
    # Isso garante que todas as classes anteriores também terão tentado o mesmo número de seeds.
    last_class_in_list = CLASSES[-1] if CLASSES else None

    # O loop principal continua até que a última classe da lista atinja o número de seeds.
    while last_class_in_list is not None and len(successful_seeds_per_class[last_class_in_list]) < NUMBER_OF_SEEDS:
        print(f"\n--- Processando Seed: {current_seed} ---")
        
        # Itera sobre as classes para a seed atual
        for target_label in CLASSES:
            
            # Se esta classe já atingiu o número necessário de seeds, pule para a próxima.
            if len(successful_seeds_per_class[target_label]) >= NUMBER_OF_SEEDS:
                print(f"Classe {target_label} já possui {NUMBER_OF_SEEDS} seeds bem-sucedidas. Pulando.")
                continue

            # Flag para rastrear se a seed foi bem-sucedida para a classe atual
            # em todas as escalas de ruído.
            seed_successful_for_this_class = True
            
            # O bloco try/except agora envolve a lógica para uma única combinação (seed, classe).
            # Uma falha aqui não afetará outras classes para a mesma seed.
            try:
                # Itera sobre as escalas de ruído para a combinação (seed, classe)
                for NOISE_SCALE in NOISE_SCALES:
                    print(f"  Executando para Classe: {target_label}, Ruído: {NOISE_SCALE}, Seed: {current_seed}")
                    np.random.seed(current_seed)
                    
                    np.random.shuffle(data)
                    m, n = data.shape

                    X_train = data[:, 1:].T / 255.0
                    Y_train = data[:, 0]
                    num_clients = 5

                    clients_data = split_data(X_train, Y_train, num_clients)
                    global_model = init_params()
                    attacked_client_idx = np.argmin([len(Y) for _, Y in clients_data])
                    client_data = clients_data[attacked_client_idx]
                    
                    X_client, Y_client = client_data
                    
                    # Verificação crucial: a classe alvo existe nos dados do cliente?
                    # Se não, esta combinação (seed, classe) é inválida.
                    if target_label not in np.unique(Y_client):
                        raise ValueError(f"Nenhuma imagem da classe {target_label} encontrada no cliente atacado para a seed {current_seed}")

                    W1, b1, W2, b2 = global_model
                    num_of_training_rounds = 50
                    milestone_for_images = 50

                    for round_num in range(num_of_training_rounds):
                        client_models = []
                        for client_id, (X_client_loop, Y_client_loop) in enumerate(clients_data):
                            weights = local_train(X_client_loop, Y_client_loop, W1, b1, W2, b2, 5, alpha=0.1)
                            noisy_weights = apply_local_differential_privacy(weights, NOISE_SCALE)
                            client_models.append(noisy_weights)

                        W1, b1, W2, b2 = aggregate(client_models)
                        global_model_accuracy = evaluate_model(X_client, Y_client, W1, b1, W2, b2)
                        client_params = client_models[attacked_client_idx]

                        loss_reconstructed_image, loss_images = reconstruct_via_loss(*client_params, target_label, client_data)
                        naive_reconstructed_image, naive_images = invert_model_naive(*client_params, target_label, client_data)
                        loss_reconstructed_image = loss_reconstructed_image.reshape(28, 28) * 255
                        naive_reconstructed_image = naive_reconstructed_image.reshape(28, 28) * 255

                        matching_indices = np.where(Y_client == target_label)[0]
                        # A verificação no início do `try` já garante que `matching_indices` não estará vazio.

                        best_ssim_loss, best_metrics_loss = -1, None
                        for idx in matching_indices:
                            candidate_image = X_client[:, idx].reshape(28, 28) * 255
                            mse_val, psnr_val, ssim_val = compute_metrics(candidate_image, loss_reconstructed_image)
                            if ssim_val > best_ssim_loss:
                                best_ssim_loss = ssim_val
                                best_metrics_loss = (mse_val, psnr_val, ssim_val)
                        mse_loss, psnr_loss, ssim_loss = best_metrics_loss

                        best_ssim_naive, best_metrics_naive = -1, None
                        for idx in matching_indices:
                            candidate_image = X_client[:, idx].reshape(28, 28) * 255
                            mse_val, psnr_val, ssim_val = compute_metrics(candidate_image, naive_reconstructed_image)
                            if ssim_val > best_ssim_naive:
                                best_ssim_naive = ssim_val
                                best_metrics_naive = (mse_val, psnr_val, ssim_val)
                        mse_naive, psnr_naive, ssim_naive = best_metrics_naive

                        if round_num == 0: display_matching_images(*loss_images)

                        if (round_num + 1) % milestone_for_images == 0:
                            results_df = display_images(*loss_images, round_num + 1, global_model_accuracy, "loss", results_df, NOISE_SCALE, mse=mse_loss, psnr=psnr_loss, ssim_val=ssim_loss, seed=current_seed)
                            results_df = display_images(*naive_images, round_num + 1, global_model_accuracy, "naive", results_df, NOISE_SCALE, mse=mse_naive, psnr=psnr_naive, ssim_val=ssim_naive, seed=current_seed)
                
                # Se o loop de NOISE_SCALES terminar sem exceções, a seed é marcada como bem-sucedida para esta classe.
            
            except Exception as e:
                # Se ocorrer um erro em qualquer ponto para esta (seed, classe), marque como falha.
                seed_successful_for_this_class = False
                print(f"\n[ERRO] A Seed {current_seed} falhou para a Classe {target_label}. Detalhe: {e}")
                print("       Pulando para a próxima classe/seed. Nenhum resultado foi salvo para esta combinação.")
                # Importante: Como a falha aconteceu, precisamos garantir que nenhum resultado parcial
                # desta combinação (seed, classe) seja salvo.
                if not results_df.empty:
                    # Remove linhas que possam ter sido adicionadas antes da falha
                    results_df = results_df[~((results_df['Seed'] == current_seed) & (results_df['Label'] == target_label))]
            
            # Após tentar todas as escalas de ruído, atualiza o contador de sucesso
            if seed_successful_for_this_class:
                successful_seeds_per_class[target_label].add(current_seed)
                print(f"--- Seed {current_seed} CONCLUÍDA com sucesso para a Classe {target_label}. Total para esta classe: {len(successful_seeds_per_class[target_label])}/{NUMBER_OF_SEEDS} ---")
                # Salva o progresso após cada combinação bem-sucedida de (seed, classe)
                results_df.to_csv("new_results_with_seed.csv", index=False)
        
        # Passa para a próxima seed para tentar
        current_seed += 1
        
        # Medida de segurança para evitar loops infinitos se for impossível encontrar seeds válidas
        if current_seed > START_SEED + (NUMBER_OF_SEEDS * 100): # Limite de 100 tentativas por seed bem-sucedida
            print("\n[AVISO] Limite de tentativas de seeds atingido. O programa será encerrado.")
            print("Verifique se os parâmetros (ex: número de clientes/amostras) permitem encontrar dados para todas as classes.")
            break

    print(f"\n--- Experimento Concluído ---")
    if last_class_in_list is not None:
        print(f"A última classe da lista ({last_class_in_list}) atingiu o número desejado de {NUMBER_OF_SEEDS} seeds.")
    else:
        print("Nenhuma classe foi especificada para o experimento.")
        
# Salva o resultado final
results_df.to_csv("new_results_with_seed.csv", index=False)
print("Resultados finais salvos. Fim do script.")
