import torch
import matplotlib.pyplot as plt

def get_causal_mask(seq_len, device='cpu', verbose=False):
    attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()  # Upper triangular matrix
    if verbose:
        print('Causal mask shape:', attn_mask.shape)
        # visualize the mask in matplotlib
        plt.imshow(attn_mask.cpu(), cmap='gray', aspect='auto')
        plt.title('Causal Mask (White = Blocked, Black = Allowed)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Sequence Length')
        plt.show()
    return attn_mask

def get_padding_mask(decoder_inputs, pad_idx, device='cpu', verbose=False):
    pad_mask = torch.zeros(decoder_inputs.size(0), decoder_inputs.size(1), dtype=torch.bool, device=device)
    pad_mask[(decoder_inputs == pad_idx)] = True
    if verbose:
        print('Padding mask shape:', pad_mask.shape)
        # visualize the mask in matplotlib
        plt.imshow(pad_mask.cpu(), cmap='gray', aspect='auto')
        plt.title('Padding Mask (White = Blocked, Black = Allowed)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Batch Size')
        plt.show()
    return pad_mask