import torch
import torch.nn as nn
import unidecode
import string
import random


num_epochs = 2000
print_every = 100
plot_every = 10

chunk_len = 200

hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
lr = 0.002

# import 했던 string에서 출력가능한 문자들을 다 불러옵니다. 
all_characters = string.printable

# 출력가능한 문자들의 개수를 저장해놓습니다.
n_characters = len(all_characters)

file = unidecode.unidecode(open('./data/text/shakespeare.txt').read())
file_len = len(file)

# 이 함수는 텍스트 파일의 일부분을 랜덤하게 불러오는 코드입니다.
def random_chunk():
    # (시작지점 < 텍스트파일 전체길이 - 불러오는 텍스트의 길이)가 되도록 시작점과 끝점을 정합니다.
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

# 문자열을 받았을때 이를 인덱스의 배열로 바꿔주는 함수입니다.
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

# 랜덤한 텍스트 chunk를 불러와서 이를 입력과 목표값을 바꿔주는 함수입니다.
# 예를 들어 pytorch라는 문자열이 들어오면 입력은 pytorc / 목표값은 ytorch 가 됩니다.
def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        
    
    def forward(self, input, hidden, cell):
        out = self.encoder(input.view(1,-1))
        out,(hidden,cell) = self.rnn(out,(hidden,cell))
        out = self.decoder(out.view(batch_size,-1))
        return out,hidden,cell

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        cell = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        return hidden,cell     
 
model = RNN(n_characters, embedding_size, hidden_size, n_characters, num_layers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 임의의 문자(start_str)로 시작하는 길이 200짜리 모방 글을 생성하는 코드입니다.
def test():
    start_str = "b"
    inp = char_tensor(start_str)
    hidden,cell = model.init_hidden()
    x = inp.to(device)
    hidden = hidden.to(device)
    cell = cell.to(device)

    print(start_str,end="")
    for i in range(200):
        output,hidden,cell = model(x,hidden,cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char,end="")

        x = char_tensor(predicted_char).to(device)

for i in range(num_epochs):
    inp,label = random_training_set()
    hidden,cell = model.init_hidden()
    inp = inp.to(device)
    label = label.to(device)
    hidden = hidden.to(device)
    cell = cell.to(device)

    loss = torch.tensor([0]).type(torch.FloatTensor).to(device)
    optimizer.zero_grad()
    for j in range(chunk_len-1):
        x  = inp[j].to(device)
        y_ = label[j].unsqueeze(0).type(torch.LongTensor).to(device)
        y,hidden,cell = model(x,hidden,cell)
        loss += loss_func(y,y_)

    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print("\n",(loss/chunk_len).cpu().detach().numpy(),"\n")
        test()
        print("\n","="*100)