using PyPlot
using LinearAlgebra, Random


"""
Gradient ascent to train a 2-layers ReLU neural net for the soft-min loss
INPUT: X (training input), Y (training output), m (nb neurons), both: training both layers or just the output
OUTPUT: Ws (training trajectory)
"""
function twonet(X, Y, m, stepsize, niter; both=true) 
    (n,d) = size(X)
    W_init = randn(m, d+1)
    if !both
        W_init[:,end] .= 0
    end

    W     = copy(W_init)
    Ws    = zeros(m, d+1, niter) # store optimization path
    loss  = zeros(niter)
    margins = zeros(niter)
    betas = zeros(niter)

    for iter = 1:niter
        Ws[:,:,iter] = W # store of the weights of iteration iter in Ws 
        act  =  max.( W[:,1:end-1] * X', 0.0) # (size m × n) # relu activation 
        out  =  (1/m) * sum( W[:,end] .* act , dims=1) # (size 1 × n)
        perf = Y .* out[:]
        margin = minimum(perf)
        temp = exp.(margin .- perf) # stabilization
        gradR = temp .* Y ./ sum(temp)' # size n
        grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  ))  # (size m × d) 
        grad_w2 = act * gradR  # size m
        
        if both
            grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
            betas[iter] = sum(W.^2)/m # single number put into betas[iter]
            loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
            margins[iter] = margin/betas[iter]
            W = W + stepsize * grad/(sqrt(iter+1))
            
         else 
            grad = cat(zeros(m,d), grad_w2, dims=2) # size (m × d+1)
            betas[iter] = maximum([1,sqrt(sum(W[:,end].^2)/m)])
            loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
            margins[iter] = margin/(sqrt(sum(W[:,end].^2))/m)
            W = W + betas[iter] * stepsize * grad /(sqrt(iter+1))
        end
    end
    Ws, loss, margins, betas
end


"Coordinates of the 2d cluster centers, p is k^2 the number of clusters"
function cluster_center(p,k)
    p1 = mod.(p .- 1,k) .+ 1
    p2 = div.(p .- 1,k) .+ 1
    Delta = 1/(3k-1)
    x1 =  Delta*(1 .+ 3*(p1 .- 1)) .- 1/2
    x2 =  Delta*(1 .+ 3*(p2 .- 1)) .- 1/2
    return x1,x2
end

"""
Plot the classifier for a test case, comparing training both or output layer
"""
function illustration(k, n, m; stepsize= 0.5, niter=100000, name="decision")
# data distribution
sd = 0 # number of spurious dimensions
Delta = 1/(3k-1) # interclass distance
A = ones(k^2) # cluster affectation get k^2 ones
A[randperm(k^2)[1:div(k^2,2)]] .= -1 # then set half of them to -1 

# sample from it
P = rand(1:k^2,n) # cluster label
T = 2π*rand(n)  # shift angle
R = Delta*rand(n) # shift magnitude
X = cat(ones(n), cluster_center(P,k)[1] .+ R .* cos.(T),cluster_center(P,k)[2] + R .* sin.(T), (rand(n,sd) .- 1/2), dims=2)
Y = A[P]

# train neural network
Ws1, loss1, margins1, betas1 = twonet(X, Y, m, stepsize, niter; both=true)
print("done training 1")
Ws2, loss2, margins2, betas2 = twonet(X, Y, m, stepsize, niter; both=false)
print("done training 2")

# plots
X1 = X[(Y .== 1),:]
X2 = X[(Y .== -1),:]
    
figure(figsize=[2.5,2.5])
    println("in here 1")
    plot(X1[:,2],X1[:,3],"+r")
    plot(X2[:,2],X2[:,3],"_b")
    plot(cluster_center(1:k^2,k)[1],cluster_center(1:k^2,k)[2],"ok")
    axis("equal");#axis("off")
    xticks([], []); yticks([], [])
    savefig(name * "setting.pdf",bbox_inches="tight")
    
figure(figsize=[2.5,2.5])
    println("in here 2")
    f1(x1,x2,t) = (1/m) * sum( Ws1[:,end,t] .* max.( Ws1[:,1:3,t] * [1;x1;x2], 0.0)) # (size 1 × n)
    xs = -0.8:0.01:0.8
    # size(Ws1, 3) means the third dimension = niter
    tab1 = [f1(xs[i],xs[j],size(Ws1,3)) for i=1:length(xs), j=1:length(xs)] 
    pcolormesh(xs', xs, tanh.(1000*tab1'),cmap="coolwarm",shading="gouraud",vmin=-1.0,vmax=1.0,edgecolor="face") # does the colour split thing 
    xs = -0.8:0.005:0.8
    tab1 = [f1(xs[i],xs[j],size(Ws1,3)) for i=1:length(xs), j=1:length(xs)]
    contour(xs', xs, tanh.(1000*tab1'),levels =0, colors="k",antialiased = true,linewidths=2) # draws a black line where the division is
    plot(X1[:,2],X1[:,3],"+k")
    plot(X2[:,2],X2[:,3],"_k")
    axis("equal");axis("off");
    savefig(name * "_both.pdf",bbox_inches="tight")
    
 figure(figsize=[2.5,2.5])
    f2(x1,x2,t) = (1/m) * sum( Ws2[:,end,t] .* max.( Ws2[:,1:3,t] * [1;x1;x2], 0.0)) # (size 1 × n)
    xs = -0.8:0.01:0.8
    tab2 = [f2(xs[i],xs[j],size(Ws2,3)) for i=1:length(xs), j=1:length(xs)]
    pcolormesh(xs', xs, tanh.(1000*tab2'),cmap="coolwarm",shading="gouraud",vmin=-1.0,vmax=1.0,edgecolor="face")
    xs = -0.8:0.005:0.8
    tab2 = [f2(xs[i],xs[j],size(Ws2,3)) for i=1:length(xs), j=1:length(xs)]
    contour(xs', xs, tanh.(1000*tab2'),levels =0, colors="k",antialiased = true,linewidths=2)
    plot(X1[:,2],X1[:,3],"+k")
    plot(X2[:,2],X2[:,3],"_k")
    axis("equal");axis("off");
    savefig(name *"_testing_output.pdf",bbox_inches="tight")
end

function main()
    print("hello")
    Random.seed!(1)
    k = 4
    n = 100
    m = 500

    illustration(k, n, m; niter=1000, name="classifier") # 10 minutes with 300000 iterations
end

main()


