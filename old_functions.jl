
# function train_network_both_single(k, n, m, t; stepsize=0.5)
#     print("in here")
#     # data distribution
#     sd = 0 # number of spurious dimensions
#     Delta = 1/(3k-1) # interclass distance
#     A = ones(k^2) # cluster affectation
#     A[randperm(k^2)[1:div(k^2,2)]] .= -1

#     println("done")

#     # sample from it
#     P = rand(1:k^2,n) # cluster label
#     T = 2π*rand(n)  # shift angle
#     R = Delta*rand(n) # shift magnitude
#     X = cat(ones(n), cluster_center(P,k)[1] .+ R .* cos.(T),cluster_center(P,k)[2] + R .* sin.(T), (rand(n,sd) .- 1/2), dims=2)
#     Y = A[P]
#     println("done")

#     Ws_both, loss_both, margins_both, betas_both = twonet(X, Y, m, stepsize, t; both=true)
#     println("done")
#     Ws_single, loss_single, margins_single, betas_single = twonet(X, y, m, stepsize, t; both=false)
#     println("done")
#     return X, Y, Ws_both, loss_both, margins_both, betas_both, Ws_single, loss_single, margins_single, betas_single
# end


# function plot_data(X, Y, Ws, k; save_flag=false, name="data_plot")
#     X1 = X[(Y .== 1),:]
#     X2 = X[(Y .== -1),:]
#     figure(figsize=[2.5,2.5])
#         plot(X1[:,2],X1[:,3],"+r")
#         plot(X2[:,2],X2[:,3],"_b")
#         plot(cluster_center(1:k^2,k)[1],cluster_center(1:k^2,k)[2],"ok")
#         axis("equal");#axis("off")
#         xticks([], []); yticks([], [])
#         if save_flag
#             savefig(name * ".pdf",bbox_inches="tight")
#         end
#     return figure
# end

# function plot_margins_original(X, Y, Ws; save_flag=false, name="margin_plot")
#     X1 = X[(Y .== 1),:]
#     X2 = X[(Y .== -1),:]
#     figure(figsize=[2.5,2.5])
#         f(x1, x2, t) = (1/m) * sum( Ws[:,end,t] .* max.( Ws[:,1:3,t] * [1;x1;x2], 0.0)) # (size 1 × n)
#         xs = -0.8:0.01:0.8
#         tab1 = [f(xs[i],xs[j],size(Ws,3)) for i=1:length(xs), j=1:length(xs)]
#         pcolormesh(xs', xs, tanh.(1000*tab1'),cmap="coolwarm",shading="gouraud",vmin=-1.0,vmax=1.0,edgecolor="face")
#         xs = -0.8:0.005:0.8
#         tab1 = [f(xs[i],xs[j],size(Ws,3)) for i=1:length(xs), j=1:length(xs)]
#         contour(xs', xs, tanh.(1000*tab1'),levels =0, colors="k",antialiased = true,linewidths=2)
#         plot(X1[:,2],X1[:,3],"+k")
#         plot(X2[:,2],X2[:,3],"_k")
#         axis("equal");axis("off");
#         if save_flag
#             savefig(name * "_both.pdf",bbox_inches="tight")
#         end
#     return figure
# end 

# X, Y, Ws_single, loss_single, margins_single, betas_single, Ws_both, loss_both, margins_both, betas_both = train_network_both_single(k, n, m, t; stepsize=0.5)

# fig = figure(figsize=(5,20))
# subplot(411)
# plot1 = plot_data(X, Y, Ws_both, k; save_flag=save, name="data_dist_both_layer")
# axis("tight")
# PyPlot.title("Data Dist Both")

# subplot(412)
# plot2 = plot_data(X, Y, Ws_single, k; save_flag=save, name="data_dist_single_layer")
# axis("tight")
# PyPlot.title("Data Dist Single")

# subplot(413)
# plot3 = plot_margins(X, Y, Ws_both; save_flag=save, name="margins_both")
# axis("tight")
# PyPlot.title("Margins Both")

# subplot(414)
# plot4 = plot_margins(X, Y, Ws_single; save_flag=save, name="margins_single")
# axis("tight")
# PyPlot.title("Margins Single")

# PyPlot.suptitle("data dist both, data dist single, margins both, margins single")
# gcf() # Needed for IJulia to plot inline

# function get_X_Y(k, n, m, t)
#     # data distribution
#     sd = 0 # number of spurious dimensions
#     Delta = 1/(3k-1) # interclass distance
#     A = ones(k^2) # cluster affectation
#     A[randperm(k^2)[1:div(k^2,2)]] .= -1

#     # sample from it
#     P = rand(1:k^2,n) # cluster label
#     T = 2π*rand(n)  # shift angle
#     R = Delta*rand(n) # shift magnitude
#     X = cat(ones(n), cluster_center(P,k)[1] .+ R .* cos.(T),cluster_center(P,k)[2] + R .* sin.(T), (rand(n,sd) .- 1/2), dims=2)
#     Y = A[P]
#     return X, Y
# end

# function twonet(X, Y, m, stepsize, niter; both=true;) 
#     (n,d) = size(X)
#     W_init = randn(m, d+1)
#     if !both
#         W_init[:,end] .= 0
#     end

#     W     = copy(W_init)
#     Ws    = zeros(m, d+1, niter) # store optimization path
#     loss  = zeros(niter)
#     margins = zeros(niter)
#     betas = zeros(niter)

#     for iter = 1:niter
#         Ws[:,:,iter] = W
#         act  =  max.( W[:,1:end-1] * X', 0.0) # (size m × n)
#         out  =  (1/m) * sum( W[:,end] .* act , dims=1) # (size 1 × n)
#         perf = Y .* out[:]
#         margin = minimum(perf)
#         temp = exp.(margin .- perf) # stabilization
#         gradR = temp .* Y ./ sum(temp)' # size n
#         grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  ))  # (size m × d) 
#         grad_w2 = act * gradR  # size m
        
#         if both
#             grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
#             betas[iter] = sum(W.^2)/m
#             loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
#             margins[iter] = margin/betas[iter]
#             W = W + stepsize * grad/(sqrt(iter+1))
            
#          else 
#             grad = cat(zeros(m,d), grad_w2, dims=2) # size (m × d+1)
#             betas[iter] = maximum([1,sqrt(sum(W[:,end].^2)/m)])
#             loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
#             margins[iter] = margin/(sqrt(sum(W[:,end].^2))/m)
#             W = W + betas[iter] * stepsize * grad /(sqrt(iter+1))
#         end
#     end
#     Ws, loss, margins, betas
# end