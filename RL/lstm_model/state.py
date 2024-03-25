from agent_lstm import *
from RL.lstm_model.lstm_guesser import *



class State(nn.Module):

    def __init__(self, features_size, embedding_dim, device):
        super(State, self).__init__()
        self.device = device
        self.features_size = features_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTMCell(input_size=self.embedding_dim *2 , hidden_size=self.embedding_dim *2 )
        self.initial_c = nn.Parameter(torch.randn(1, self.embedding_dim *2 ), requires_grad=True).to(
            device=self.device)
        self.initial_h = nn.Parameter(torch.randn(1, self.embedding_dim *2), requires_grad=True).to(
            device=self.device)
        self.reset_states()

    def reset_states(self):
        self.lstm_h = (torch.zeros(1, self.embedding_dim *2) + self.initial_h).to(device=self.device)
        self.lstm_c = (torch.zeros(1, self.embedding_dim *2) + self.initial_c).to(device=self.device)

    def forward(self, question_encode, answer):
        answer_vec = torch.unsqueeze(torch.ones(self.embedding_dim) * answer, 0)
        question_embedding = question_encode.to(device=self.device)
        answer_vec = answer_vec.to(device=self.device)
        x = torch.cat((question_embedding,answer_vec), dim=1)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h, self.lstm_c))
        return self.lstm_h.data.cpu().numpy()
