using LinearAlgebra
import PyPlot
plt = PyPlot

function classical_gs_naive(A)
    m, n = size(A)
    Q = copy(A)
    R = zeros(n, n)
    for j = 1:n
        v = Q[:, j] # j-th column of A
        for i = 1:j-1 # Project out the components on the previous vectors
            w = Q[:, i]
            R[i, j] = w' * A[:, j]
            v .-= R[i, j] * w
        end
        R[j, j] = norm(v) # Normalize
        Q[:, j] .= v / R[j, j]
    end
    return Q, R
end

function modified_gs(A)
    m, n = size(A)
    Q = copy(A)
    R = zeros(n, n)
    for j = 1:n
        v = Q[:, j] # j-th column of Q to be normalized
        R[j, j] = norm(v) # Normalize
        Q[:, j] .= v / R[j, j]
        # Project out the components of the remaining vectors
        w = Q[:, j]
        for i = j+1:n
            R[j, i] = w' * A[:, i]
            Q[:, i] .-= R[j, i] * w
        end
    end
    return Q, R
end


function test(m,n,rate)
    # Construct a matrix with decaying singular values
    U, _ = qr(rand(m, n))
    V, _ = qr(rand(n, n))
    σs = [rate^i for i = 1:min(m, n)]
    Σ = diagm(σs)
    A = U * Σ * V'
    @show cond(A)

    println("\nFactoring times:")
    @time Q1, R1 = classical_gs_naive(A)
    @time Q2, R2 = modified_gs(A)
    @time QR = qr(A)
    println("\nFactoring errors:")
    @show norm(Q1' * Q1 - I)
    @show norm(Q2' * Q2 - I)
    @show norm(QR.Q' * QR.Q - I)
    @show norm(Q1 * R1 - A)
    @show norm(Q2 * R2 - A)
    @show norm(QR.Q * QR.R - A)

    plt.figure()
    plt.semilogy(diag(R1), label="classical")
    plt.semilogy(diag(R2), label="modified")
    plt.semilogy(abs.(diag(QR.R)), label="julia qr")
    plt.title("Diagonal of R")
    plt.axhline(sqrt(eps()), color="red", linestyle="--")
    plt.axhline(eps(), color="black", linestyle="--")
    plt.legend()
end
