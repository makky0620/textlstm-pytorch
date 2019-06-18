import torch
import torch.nn as nn

dtype = torch.FloatTensor

class TextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TextLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedded = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        embedded = self.embedded(input)
        outputs, hidden = self.lstm(embedded)
        output = outputs[-1]
        output = self.fc(output)
        return output

    def init_hidden(self):
        return torch.zeros(1, 3, self.hidden_size)

