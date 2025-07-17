import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from PIL import Image, ImageFile

class CurriculumDataLoader:
    def __init__(self, original_dataset, generated_dataset, batch_size, total_steps, mix_ratio, num_workers):
        """
        Custom DataLoader for Curriculum Learning.
        
        Args:
            original_dataset (Dataset): The original dataset.
            generated_dataset (Dataset): The generated dataset.
            batch_size (int): The batch size.
            total_steps (int): The total number of training steps.
            increase_step (int): Step interval after which the generated data ratio increases.
        """
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.mix_ratio = mix_ratio
        self.current_step = 0
        self.original_loader = DataLoader(self.original_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.generated_loader = DataLoader(self.generated_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.original_iter = iter(self.original_loader)
        self.generated_iter = iter(self.generated_loader)
    
    def __iter__(self):
        return self

    def __next__(self):
        """Return the next batch of data with a ratio of original and generated data."""
        # Calculate the ratio of generated data based on the current step
        # generated_ratio = min(self.current_step // self.increase_step, 1.0)
        generated_ratio = min(self.mix_ratio, self.current_step / (self.total_steps / self.mix_ratio)) 
        num_generated = int(self.batch_size * generated_ratio)
        num_original = self.batch_size - num_generated

        # Sample data from both original and generated datasets
        original_data = self._get_next_batch(self.original_iter, self.original_loader, num_original)
        generated_data = self._get_next_batch(self.generated_iter, self.generated_loader, num_generated)
        
        # Concatenate original and generated data
        if generated_data != None:
            data = torch.cat([original_data[0], generated_data[0]], dim=0)
            labels = torch.cat([original_data[1], generated_data[1]], dim=0)
        else:
            data, labels = original_data

        self.current_step += 1
        return data, labels

    def _get_next_batch(self, loader_iter, loader, num_samples):
        """Helper function to get the next batch from a dataset iterator."""
        batch_data = []
        batch_labels = []
        try:
            data, labels = next(loader_iter)
            batch_data.append(data[:num_samples])
            batch_labels.append(labels[:num_samples])
        except StopIteration:
            # If we run out of data in the iterator, reset it
            loader_iter = iter(loader)
            data, labels = next(loader_iter)
            batch_data.append(data[:num_samples])
            batch_labels.append(labels[:num_samples])

        # Concatenate all collected data and labels
        return torch.cat(batch_data, dim=0), torch.cat(batch_labels, dim=0)

    def __len__(self):
        return self.total_steps
