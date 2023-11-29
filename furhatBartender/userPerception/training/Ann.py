import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Assets import load_data
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def accuracy(G, Y):
    return (G.argmax(dim=1) == Y).float().mean()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # there is 20,1 as input units
        # should be 7 out, 7 emotions
        U1 = 60  # number of hidden units
        U2 = 30
        U3 = 15

        self.W1 = nn.Parameter(0.1 * torch.randn(20, U1))
        self.b1 = nn.Parameter(torch.ones(U1) / 10)
        self.W2 = nn.Parameter(0.1 * torch.randn(U1, U2))
        self.b2 = nn.Parameter(torch.ones(U2) / 10)
        self.W3 = nn.Parameter(0.1 * torch.randn(U2, U3))
        self.b3 = nn.Parameter(torch.ones(U3) / 10)
        self.W4 = nn.Parameter(0.1 * torch.randn(U3, 7))
        self.b4 = nn.Parameter(torch.ones(7) / 10)

    def forward(self, X):
        Q1 = F.relu(X.mm(self.W1) + self.b1)
        Q1 = F.dropout(Q1, 0.5)
        Q2 = F.relu(Q1.mm(self.W2) + self.b2)
        Q2 = F.dropout(Q2, 0.5)
        Q3 = F.relu(Q2.mm(self.W3) + self.b3)
        Q3 = F.dropout(Q3, 0.25)
        Z = Q3.mm(self.W4) + self.b4

        return Z


def main():
    test_accuracy = []
    test_crossentropy = []
    test_iter = []
    train_accuracy = []
    train_crossentropy = []
    train_iter = []
    le = LabelEncoder()
    train, validation, test = load_data()
    le.fit(train[1])

    X_train = torch.FloatTensor(train[0].values)
    X_val = torch.FloatTensor(validation[0].values)
    y_train = torch.LongTensor(le.transform(train[1]))
    y_val = torch.LongTensor(le.transform(validation[1]))
    y_test = torch.LongTensor(le.transform(test[1]))
    X_test = torch.FloatTensor(test[0].values)

    net = Net()
    learningrate = 0.0008
    optimizer = optim.Adam(net.parameters(), lr=learningrate)

    total_iterations = 6400  # total number of iterations
    t = 0  # current iteration

    while True:
        optimizer.zero_grad()
        net.train()
        y_hat = net(X_train)
        loss = F.cross_entropy(y_hat, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if t % 10 == 0:
                net.eval()
                train_crossentropy.append(loss.item())
                train_accuracy.append(accuracy(y_hat, y_train).item())
                train_iter.append(t)

            if t % 100 == 0:
                net.eval()
                G = net(X_val)
                test_crossentropy.append(F.cross_entropy(G, y_val).item())
                test_accuracy.append(accuracy(G, y_val).item())
                test_iter.append(t)
                print(
                    f"Step {t:5d}: train accuracy {100 * train_accuracy[-1]:6.2f}% "
                    f"train cross-entropy {train_crossentropy[-1]:5.2f}  "
                    f"test accuracy {100 * test_accuracy[-1]:6.2f}% "
                    f"test cross-entropy {test_crossentropy[-1]:5.2f}"
                )
        t += 1
        if t > total_iterations:
            break
    # need to save the ebst model with respect to validation set and test it on my test set
    # might also ahve to do something like dropout to not overfit as badly
    # best am is around 62% so worse than some

    plt.plot(train_iter, train_accuracy, "b-", label="Training data (mini-batch)")
    plt.plot(test_iter, test_accuracy, "r-", label="Test data")
    plt.xlabel("Iteration")
    plt.ylabel("Prediction accuracy")
    plt.ylim([max(1 - (1 - test_accuracy[-1]) * 2, 0), 1])
    plt.title("Prediction accuracy")
    plt.grid(True)
    plt.legend(loc="best")
    print(max(test_accuracy), test_accuracy.index(max(test_accuracy)))
    plt.show()
    net.eval()
    preds = net(X_test)
    print(accuracy(preds, y_test))


if __name__ == "__main__":
    main()
