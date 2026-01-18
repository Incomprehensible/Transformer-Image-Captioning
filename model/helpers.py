import torch
import matplotlib.pyplot as plt

# size: (seq_len, seq_len)
def get_causal_mask(seq_len, device='cpu', verbose=False):
    attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, requires_grad=False), diagonal=1).bool()  # Upper triangular matrix
    if verbose:
        print('Causal mask shape:', attn_mask.shape)
        # visualize the mask in matplotlib
        plt.imshow(attn_mask.cpu(), cmap='gray', aspect='auto')
        plt.title('Causal Mask (White = Blocked, Black = Allowed)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Sequence Length')
        plt.show()
    return attn_mask

# shape: (batch_size, seq_len)
def get_padding_mask(decoder_inputs, pad_idx, device='cpu', verbose=False):
    pad_mask = torch.zeros(decoder_inputs.size(0), decoder_inputs.size(1), dtype=torch.bool, device=device, requires_grad=False)
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

def get_causal_and_padding_mask(decoder_inputs, pad_idx, seq_len, device='cpu', verbose=False):
    causal_mask = get_causal_mask(seq_len, device=device, verbose=verbose)  # (seq_len, seq_len)
    padding_mask = get_padding_mask(decoder_inputs, pad_idx, device=device, verbose=verbose)  # (batch_size, seq_len)

    # Combine masks: expand padding mask to (batch_size, 1, seq_len) for broadcasting
    combined_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)  # (batch_size, seq_len, seq_len)

    if verbose:
        print('Combined mask shape:', combined_mask.shape)
        # visualize the combined mask in matplotlib
        plt.imshow(combined_mask[0].cpu(), cmap='gray', aspect='auto')
        plt.title('Combined Causal and Padding Mask (White = Blocked, Black = Allowed)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Sequence Length')
        plt.show()

    return combined_mask