import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
import cv2
import os

results_data = []
# Load or create the DataFrame with proper columns
results_path = "new_results_with_seed.csv"
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
else:
    results_df = pd.DataFrame(columns=[
        'Label', 'Round', 'Method', 'Noise Scale',
        'Accuracy', 'Loss', 'MSE', 'PSNR', 'SSIM', 'Seed'
    ])

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
    return expZ / expZ.sum(axis=0, keepdims=True)


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    return one_hot_Y.T

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def split_data(X, Y, num_clients, samples_per_client=50):
    clients = []
    for _ in range(num_clients):
        indices = np.random.choice(X.shape[1], samples_per_client, replace=False)
        X_client = X[:, indices]
        Y_client = Y[indices]
        clients.append((X_client, Y_client))
    return clients

def init_params():
    W1 = np.random.randn(10, 784) * 0.01 
    b1 = np.zeros((10, 1))                
    W2 = np.random.randn(10, 10) * 0.01  
    b2 = np.zeros((10, 1))                
    return W1, b1, W2, b2

def local_train(X, Y, W1, b1, W2, b2, iterations, alpha):
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2

def aggregate(models):
    num_clients = len(models)
    aggregated = [np.mean([model[param] for model in models], axis=0) for param in range(len(models[0]))]
    return tuple(aggregated)

def evaluate_model(X, Y, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    accuracy = np.mean(predictions == Y) * 100
    return accuracy

# Not used --- Implemented in main
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

def preprocess_lpips(image):
    image = cv2.resize(image, (64, 64)) 
    image = np.stack([image] * 3, axis=-1) 
    image = (image / 255.0) * 2 - 1 
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

def compute_metrics(real_image, reconstructed_image):
    mse = np.mean((real_image - reconstructed_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    ssim_value = ssim(real_image, reconstructed_image, data_range=255)
    real_tensor = preprocess_lpips(real_image)
    reconstructed_tensor = preprocess_lpips(reconstructed_image)

    return mse, psnr, ssim_value

def localized_perturbation(X_sample, perturbation_size=50, sigma=0.05):
    perturbation = np.zeros_like(X_sample)
    active_pixels = np.random.choice(784, perturbation_size, replace=False) 
    perturbation[active_pixels] = np.random.normal(0, sigma, size=(perturbation_size, 1))
    return perturbation

def display_images(images, losses, matching_images, labels, iterations, round_num,
                   global_model_accuracy, method, results_df,
                   mse=None, psnr=None, ssim_val=None, seed=None):
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    
    # Display the last image without title
    for i, img in enumerate(images):
        if i == len(images)-1:
            axes.imshow(img, cmap='gray')
            axes.axis('off')
    
    # Save the image
    filename = f"{labels[-1]}/{NOISE_SCALE} noise/{method} Round {round_num}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    new_row = {
        'Label': labels[-1],
        'Round': round_num,
        'Method': method,
        'Noise Scale': NOISE_SCALE,
        'Accuracy': global_model_accuracy,
        'Loss': losses[-1],
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_val,
        'Seed': seed
    }

    # If a DataFrame was passed, append to it
    
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df


def display_matching_images(images, losses, matching_images, labels, iterations):
    fig, axes = plt.subplots(1,len(matching_images), figsize=(15, 5))
    if len(matching_images) > 1:
        for i, img in enumerate(matching_images):
            axes[i].imshow(img.reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Real {labels[0]}")
            axes[i].axis('off')
    else:
        for i, img in enumerate(matching_images):
            axes.imshow(img.reshape(28, 28), cmap='gray')
            axes.set_title(f"Real {labels[0]}")
            axes.axis('off')
    filename = f"{labels[0]}/Matching Images.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

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
            #print(f"\n\n")
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

    #print(f"\nFinal Iteration: {iterations}")
    #print(f"Final Certainty: {certainty:.5f}")
    #print(f"Target Label: {target_label}")

    milestone_images.append(X_sample.reshape(28, 28))
    milestone_certainties.append(certainty)
    milestone_iterations.append(iterations)

    matching_indices = np.where(Y_client == target_label)[0]
    matching_images = X_client[:, matching_indices].T

    matching_images = X_client[:, matching_indices].T

    num_matching = matching_images.shape[0]
    #print(f"\nFound {num_matching} matching images for label {target_label} in the client dataset.")

    return X_sample, (milestone_images, milestone_certainties, matching_images[:10], [target_label] * len(milestone_images), milestone_iterations)

def reconstruct_via_loss(W1, b1, W2, b2, target_label, client_data, learning_rate=0.01, iterations=3000, momentum=0.9):
    #print(f"\nAttempting reconstruction for label: {target_label}")

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

        #dX_reconstructed /= np.linalg.norm(dX_reconstructed) + 1e-8 

        velocity = -learning_rate * dX_reconstructed #+ momentum * velocity 
        X_reconstructed += velocity

        X_reconstructed = np.clip(X_reconstructed, 0, 1)

        if i % 500 == 0 or i == iterations - 1:
            milestone_images.append(X_reconstructed.reshape(28, 28))
            milestone_losses.append(loss)
            milestone_iterations.append(i)
            #print(f"Iteration {i}: Loss = {loss:.5f}")

    matching_indices = np.where(Y_client == target_label)[0]
    matching_images = X_client[:, matching_indices].T

    num_matching = matching_images.shape[0]
    #print(f"\nFound {num_matching} matching images for label {target_label} in the client dataset.")

    return X_reconstructed, (milestone_images, milestone_losses, matching_images[:10], [target_label] * len(milestone_images), milestone_iterations)

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

def apply_local_differential_privacy(weights):
    """Apply LDP with fixed clipping and additive noise."""
    # clipped_weights = clip(weights, CLIP_NORM)
    # noisy_weights = [add_noise(w, NOISE_SCALE) for w in clipped_weights]
    noisy_weights = [add_noise(w, NOISE_SCALE) for w in weights]
    return noisy_weights


def add_noise(weight, noise_scale):
    noise = np.random.normal(0, noise_scale, size=weight.shape)
    return weight + noise

data = pd.read_csv("mnist_train.csv").to_numpy()
exception_counter = 0

if __name__ == "__main__":
    for seed in range(20, 22):
        for i in range(0, 10):
            ocorreu_erro_neste_i = False
            for NOISE_SCALEtime100 in range(0, 11):
                try:
                    print(f"Classe {i}, NOISE_SCALE: {NOISE_SCALEtime100/100}, seed: {seed}")
                    np.random.seed(seed)
                    NOISE_SCALE = NOISE_SCALEtime100 / 100
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
                    target_label = i

                    W1, b1, W2, b2 = global_model
                    local_models = []
                    num_of_training_rounds = 50
                    milestone_for_images = 50

                    for round_num in range(num_of_training_rounds):
                        client_models = []
                        for client_id, (X_client, Y_client) in enumerate(clients_data):
                            weights = local_train(X_client, Y_client, W1, b1, W2, b2, 5, alpha=0.1)
                            noisy_weights = apply_local_differential_privacy(weights)
                            client_models.append(noisy_weights)

                        W1, b1, W2, b2 = aggregate(client_models)
                        global_model_accuracy = evaluate_model(X_client, Y_client, W1, b1, W2, b2)
                        local_models = client_models
                        client_params = local_models[attacked_client_idx]

                        loss_reconstructed_image, loss_images = reconstruct_via_loss(*client_params, target_label, client_data)
                        naive_reconstructed_image, naive_images = invert_model_naive(*client_params, target_label, client_data)
                        loss_reconstructed_image = loss_reconstructed_image.reshape(28, 28) * 255
                        naive_reconstructed_image = naive_reconstructed_image.reshape(28, 28) * 255

                        matching_indices = np.where(Y_client == target_label)[0]
                        if len(matching_indices) == 0:
                            raise ValueError(f"Nenhuma imagem da classe {target_label} encontrada para a seed {seed}")

                        best_ssim = -1
                        best_metrics = None

                        for idx in matching_indices:
                            candidate_image = X_client[:, idx].reshape(28, 28) * 255
                            mse_val, psnr_val, ssim_val = compute_metrics(candidate_image, loss_reconstructed_image)
                            if ssim_val > best_ssim:
                                best_ssim = ssim_val
                                best_metrics = (mse_val, psnr_val, ssim_val)

                        mse_loss, psnr_loss, ssim_loss = best_metrics

                        best_ssim = -1
                        best_metrics_naive = None

                        for idx in matching_indices:
                            candidate_image = X_client[:, idx].reshape(28, 28) * 255
                            mse_val, psnr_val, ssim_val = compute_metrics(candidate_image, naive_reconstructed_image)
                            if ssim_val > best_ssim:
                                best_ssim = ssim_val
                                best_metrics_naive = (mse_val, psnr_val, ssim_val)

                        mse_naive, psnr_naive, ssim_naive = best_metrics_naive

                        if round_num == 0:
                            display_matching_images(*loss_images)

                        if (round_num + 1) % milestone_for_images == 0:
                            results_df = display_images(*loss_images, round_num + 1, global_model_accuracy, "loss", results_df,
                                                        mse=mse_loss, psnr=psnr_loss, ssim_val=ssim_loss, seed=seed)
                            results_df = display_images(*naive_images, round_num + 1, global_model_accuracy, "naive", results_df,
                                                        mse=mse_naive, psnr=psnr_naive, ssim_val=ssim_naive, seed=seed)

                            if NOISE_SCALE == 0 and i == 0 and round_num == 50:
                                results_df.to_csv("temp_new_results_with_seed.csv", index=False)

                        results_df.to_csv("new_results_with_seed.csv", index=False)

                except Exception as e:
                    # exception_counter += 1
                    print(f"\n[ERRO] Ocorreu uma exceção para (Seed={seed}, Target={i}, Noise={NOISE_SCALE}).")
                    print(f"Detalhe: {str(e)}")
                    print("Limpando entradas corrompidas e pulando para o próximo 'i'...")

                    ## AQUI: Lógica de deleção mais precisa
                    # Criamos uma condição para MANTER todas as linhas EXCETO aquelas com a seed E o label que falharam.
                    # Nota: Assumi que a coluna de label se chama 'Target'. Altere se for 'Label' ou outro nome.
                    if not results_df.empty:
                        condicao_para_manter = ~((results_df['Seed'] == seed) & (results_df['Label'] == i))
                        results_df = results_df[condicao_para_manter]
                        # Salva o dataframe limpo
                        results_df.to_csv("new_results_with_seed.csv", index=False)
                    
                    ## AQUI: Ativamos a flag e quebramos o loop do NOISE_SCALE
                    ocorreu_erro_neste_i = True
                    break # Sai do loop do NOISE_SCALEtime100

            ## AQUI: Verificamos a flag após o loop do NOISE_SCALE
            if ocorreu_erro_neste_i:
                # Se a flag for verdadeira, usamos 'continue' para pular para a próxima iteração do loop 'i'
                continue

# Final save
results_df.to_csv("new_results_with_seed.csv", index=False)
print("acabou")
