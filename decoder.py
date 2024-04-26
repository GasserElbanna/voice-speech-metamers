import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_state):
        output, (hn, cn) = self.lstm(hidden_state)
        logits = self.linear(output)
        return logits
    
    # def init_hidden(self, batch_size):
    #     # Initialize the hidden states and cell states
    #     h_0 = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
    #     c_0 = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
    #     return (h_0, c_0)

if __name__ == '__main__':
    # Example usage:
    input_size = 256
    hidden_size = 512
    num_layers = 2
    output_size = 40

    lstm_model = LSTM(input_size, hidden_size, output_size, num_layers)

    # Test forward pass
    input_data = torch.randn(4, 100, input_size)  # input tensor of shape (batch, seq_len, input_size)
    output = lstm_model(input_data)
    print("Output shape:", output.shape)  # Expected output shape: (batch_size, seq_len, output_size)
