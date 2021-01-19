import torch

#rounding to decimal places in torch
def custom_round(x,dec):
    if dec == 0:
        return x.round()
    else:
        return (x * 10**dec).round()/10**dec

#efficient implementation of squared distance calculation (faster than cdist**2)
def squared_distance(x):
    norm = (x ** 2).sum(1).view(-1, 1)
    dist_mat = (norm + norm.view(1, -1)) - 2.0 * torch.mm(x , x.t())
    return dist_mat

#code to generate random covariance matrices with a given scale
def random_covariance(D,Lmin=1,Lmax=1):
    #generate diagonal matrix with eigenvalues in the desired range (i.e. [1e-1,1e1])
    L = torch.eye(D) * (torch.rand([D])*(Lmax - Lmin) + Lmin)
    #generate a random unitary matrix (use QR decomposition to obtain)
    Q,_ = torch.qr(torch.distributions.MultivariateNormal(torch.zeros(D),torch.eye(D)).sample([D]))
    #obtain the desired covariance by multiplication:
    sigma = torch.mm(torch.mm(Q,L),Q.T)
    return sigma

#implementation of un-corrected covariance for torch
def estimate_cov(x):
    N = x.shape[0]
    m = torch.mean(x,0)
    assert N > 1
    return torch.mm((x-m).T,(x-m))/(N)

def rbf_kernel(x,N,h=-1):
    #pairwise squared euclidean distances
    dist_mat = squared_distance(x)
    #if no proper bandwidth is given
    if h < 0:
        #use the heuristic from liu&wang2016
        h = torch.median(dist_mat)/torch.log(torch.FloatTensor([N+1]))
    #rbf kernel formula
    kxx = torch.exp(-dist_mat/h)
    #vectorized form of the derivative of the rbf kernel
    dxkxx = (-torch.mm(kxx,x)+kxx.sum(1).view(-1,1)*x)*(2/h)
    return kxx, dxkxx


def linear_kernel(x,N):
    #linear kernel is defined as x * x.T + c
    kxx = torch.mm(x,x.T)+torch.ones([N,N])
    #vectorized derivative
    dkxx = (N+1) * x
    return kxx, dkxx

def svgd(x0,p,kernel = 'linear',num_iter = 1000, eta = 1e-2,history=False):
    x = x0.clone().detach()
    N = x.shape[0]
    D = x.shape[1]
    
    if history:
        x_history = torch.zeros((num_iter,N,D))
        
    for it in range(num_iter):
        #enable autograd for x
        x.requires_grad = True
        
        #calculate kernel and kernel_gradient
        if kernel == 'rbf':
            kxx,dkxx = rbf_kernel(x,N)
        elif kernel == 'linear':
            kxx,dkxx = linear_kernel(x,N)
        else:
            raise NotImplementedError('Please choose from linear or rbf kernel')
            
        #calculate the log-likelihood of x
        logpx = p.log_prob(x)
        #calculate the derivative of the log-likelihood wrt x
        logpx.sum().backward(retain_graph = True)
        dxlogpx = x.grad.clone()
        #zero x.grad for later use
        x.grad.data.zero_()    
        #calculate optimal perturbation direction
        phi = (1/N) * (torch.mm(kxx,dxlogpx) + dkxx)
        #make a step of gradient descent
        x_new = x + eta * phi
        
        if history:
            x_history[it] = x_new.clone().detach()
        x = x_new.clone().detach()
        
    if history:
        return x_history
    else:
        return x

def gpf(x0,p,num_iter=1000,eta1=0.1,eta2=0.1,history=False):
    x = x0.clone().detach()
    N = x.shape[0]
    D = x.shape[1]
    
    if history:
        x_history = torch.zeros((num_iter,N,D))
        
    for j in range(num_iter):
        #calculate score
        x.requires_grad = True
        log_px = -p.log_prob(x) 
        log_px.sum().backward(retain_graph = True)
        #compute g and g_
        g = x.grad
        g_ = torch.mean(g,0)
        #center the particles
        x_m = x - torch.mean(x,0)
        #calculate Matrix A particle-wise
        a = torch.zeros([D,D])
        for i in range(N):
            a += torch.mm(g[i].view(D,1),x_m[i].view(1,D))
        A = a/N - torch.eye(D)
        #update particles
        x_new = x - eta1*g_.view(1,D) - eta2*torch.mm(x_m,A) 
        x = x_new.clone().detach()
        
        if history:
            x_history[j] = x_new.clone().detach()
            
    if history:
        return x_history
    else:
        return x


def animate_trajectory(theta_hist,target_dist,amin,amax):
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    from matplotlib import rc
    
    rc('animation', html='html5')
    
    assert theta_hist.shape[2] == 2, 'can only animate 2D trajectories'
    
    #create the contour data for the target distribution
    a = torch.linspace(amin,amax,100)
    x,y = torch.meshgrid(a,a)
    z = torch.zeros((100,100))
    for i,ai in enumerate(a):
        for j,aj in enumerate(a):
            z[i,j] = torch.exp(target_dist.log_prob(torch.Tensor([ai,aj])))
    
    #change the trajectory to numpy
    x_hist = theta_hist.numpy()
    #initialize the figure
    fig,ax = plt.subplots()
    ax.axis('square')
    ax.set_xlim([amin,amax])
    ax.set_ylim([amin,amax])
    ax.contour(x,y,z)
    particles = ax.scatter([],[])
    plt.close()
    #animation functions
    def init():
        particles = ax.scatter([],[])
        return particles,

    def animate(i):
        particles.set_offsets(x_hist[i])
        return particles,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=x_hist.shape[0], interval=20, blit=False)
    return anim
