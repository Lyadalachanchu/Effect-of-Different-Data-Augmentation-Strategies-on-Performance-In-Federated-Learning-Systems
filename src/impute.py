import random

import numpy as np
import torch.utils.data

from vae.mnist_vae import ConditionalVae, VaeAutoencoderClassifier
from torchvision import transforms
from collections import Counter
def get_label_distribution(dataset):
    # # Get the distribution of labels in the initial dataset
    # labels = [label for _, label in dataset]
    # label_counts = Counter(labels)
    # total_count = sum(label_counts.values())
    # label_distribution = {label: count / total_count for label, count in label_counts.items()}
    # return label_distribution
    post_label_counts = [label for _, label in dataset]
    post_label_counts_counter = Counter(post_label_counts)

    # Convert Counter to array
    post_label_counts_array = np.zeros(10)
    for label, count in post_label_counts_counter.items():
        post_label_counts_array[label] += count/len(dataset)
    return post_label_counts_array

def impute_naive(k, trained_vae:VaeAutoencoderClassifier, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = trained_vae.generate_data(n_samples=k)
    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[0][image_ind], np.argmax(generated_dataset[1][image_ind].detach().numpy())))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])

def impute_cvae_naive(k, trained_cvae:ConditionalVae, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = []
    uniform_digits = [random.randint(0, 9) for _ in range(k)]
    for i in uniform_digits:
        generated_image = trained_cvae.generate_data(n_samples=1, target_label=i).squeeze(1)
        multiplier = 1.0/generated_image.max().item()
        transformed_image = torch.round(generated_image*multiplier)
        generated_dataset.append((transformed_image, i))

    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[image_ind][0], generated_dataset[image_ind][1]))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])



def impute_cvae_minority(k, trained_cvae, initial_dataset):
    # Get the label distribution from the initial dataset
    label_distribution = get_label_distribution(initial_dataset)

    # Generate dataset
    generated_dataset = []

    defecits = []
    for label, count in enumerate(label_distribution):
        defecits.append(max(label_distribution)-count)
    inverse_counts = [k*(defe/sum(defecits)) for defe in defecits]
    sum_samples_added = 0
    for label, count in enumerate(label_distribution):
        num_samples_to_generate = round(inverse_counts[label]+0.5)
        sum_samples_added += num_samples_to_generate
        print(f"num samples added: {num_samples_to_generate}")
        if num_samples_to_generate > 0:
            generated_images = trained_cvae.generate_data(n_samples=num_samples_to_generate, target_label=label)
            transformed_images = []
            for g in generated_images:
                multiplier = 1.0 / g.max().item()
                transformed_image = torch.round(g * multiplier)
                transformed_images.append(transformed_image)
                generated_dataset.append((transformed_image, label))
            # generated_dataset.extend([(image.squeeze(1), label) for image in transformed_images])

    # Convert generated dataset to the same format as impute_cvae_naive
    imputed_samples = [(sample, label) for sample, label in generated_dataset]

    return torch.utils.data.ConcatDataset([initial_dataset, imputed_samples])