# import functionality from these libraries
import numpy as np      # for efficient numerical computation
import torch            # for building computational graphs
from torch.autograd import Variable     # for automatically computing gradients of our cost with respect to what we want to optimise
import matplotlib.pyplot as plt     # for plotting absolutely anything
from mpl_toolkits.mplot3d import Axes3D     # for plotting 3D graphs

# define hyperparameters as needed
n = 2       # number of features e.g. number of windows, number of rooms
m = 500     # number of training examples, e.g. the number of windows and rooms was measured for m houses

epochs = 100    # how many times do we want to run through the data to train our model
lr = 0.05        # what proportion of our gradient do we want to update our parameters by

#create dataset and variables - built in models use one row per data point rather than one column
X = Variable(torch.rand(m, n))
Y = Variable(2*X.data[:,0] + 1.6*X.data[:,1] + 1)

# make the variables that we want to optimise
w = Variable(torch.randn((1, 2)), requires_grad=True)   # requires_true lets PyTorch know whether to make a place in
# this variable to store the derivatives of something with respect to it
b = Variable(torch.Tensor(1), requires_grad=True)     # w and b are places on our computational graph, so they are torch variables

#for plotting costs
costs = []
plt.ion()
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121, projection='3d')#start with 111

x1 = np.arange(2)
x2 = np.arange(2)
x1, x2 = np.meshgrid(x1, x2)
#ax1.plot_surface(x1, x2, 1 + 0.5*x1 + 1.6*x2)

ax1.scatter(X.data[:, 0], X.data[:, 1], Y.data[:]) #plot data points
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
x1 = np.arange(2)
x2 = np.arange(2)
x1, x2 = np.meshgrid(x1, x2)

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_xlim(0, epochs)
ax2.grid()
plt.show()

#define model class
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

#create model and define training params
m = LinearModel()
for i in m.parameters():
    print(i.data[0])


criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(m.parameters(), lr=lr)
#train model
losses = []

for epoch in range(epochs):
    y_pred = m(X)

    cost = criterion(y_pred, Y)

    costs.append(cost.data)
    print('Epoch ', epoch, ' Cost: ', cost.data[0])

    optimizer.zero_grad()
    # find rate of change of J with respect to each rg=True variable and put that in tht var.grad
    cost.backward()

    optimizer.step()
    # calling J.backwards() does not set the values in the var.grad,it adds to them
    # otherwise it will contain the cumulative history of grads

    w.data[0], b.data[0] = [i.data[0] for i in m.parameters()]
    print('w', w.data[0], 'b', b.data[0])

    #plot costs
    ax2.plot(costs, 'b')

    #y = b.data[0] + x1*w.data[0][0] + x2*w.data[0][1] #calculate hypothesis surface

    y = b.data[0] + x1 * w.data[0][0] + x2 * w.data[0][1] #calculate hypothesis surface


    s = ax1.plot_surface(x1, x2, y, color=(0, 1, 1, 0.5)) #plot surface
    #plot data points + hypothesis surface
    ax1.view_init(azim=epoch) #choose view angle of 3d plot
    fig.canvas.draw()       # draw the new stuff on the canvas (the
    s.remove() # remove the 3d surface plot object
    optimizer.step()

#use model to predict new value
v = Variable(torch.Tensor([[4, 0]]))
print('Predict [4, 0]: ', m.forward(v).data[0][0])

# save variables for which rg =True
print(m.state_dict())
torch.save(m.state_dict(), 'saved_linear_model')
#load model
#m = Model()
#m.load_state_dict(torch.load('savedmodel'))

