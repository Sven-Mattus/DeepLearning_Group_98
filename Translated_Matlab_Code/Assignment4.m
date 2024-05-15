
% read in the data
book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

% all contained unique chars
book_chars = unique(book_data);
K = length(book_chars);

% to easily go between a character and its one-hot encoding (index)
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
for i = 1:K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

function onehot = charsToOneHot(chars, K, char_to_ind)
    onehot = zeros(K, length(chars));
    for i = 1:length(chars)
        ind = char_to_ind(chars(i));
        onehot(ind,i) = 1;
    end
end

function chars = oneHotToChars(onehot, ind_to_char)
    n = size(onehot,2);
    chars = char([1, n]);
    for i = 1:n
        ind = find(onehot(:,i), 1);
        chars(1,i) = ind_to_char(ind);
    end
end

% set hyperparameters
m = 100; % dimensionality of hidden state
eta = 0.1;
seq_length = 25;

% initialize the RNN's parameters
sig = .01;
RNN = struct('b', zeros(m,1), 'c', zeros(K,1), 'U', randn(m, K)*sig, 'W', randn(m, m)*sig, 'V', randn(K, m)*sig);

function P = SoftMax(s)
    P = exp(s) ./ sum(exp(s)); 
end

function Y = SynthesizeText(RNN, h0, x0, n, char_to_ind) % x0 char
    K = size(RNN.c, 1);
    ht = h0; % size mx1
    xt = charsToOneHot(x0, K, char_to_ind);
    xiis = zeros(n, 1);
    Y = zeros(K, n);
    for t = 1:n
        at = RNN.W * ht + RNN.U * xt + RNN.b;
        ht = tanh(at);
        ot = RNN.V * ht + RNN.c;
        pt = SoftMax(ot);
        
        % sample next x
        cp = cumsum(pt);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1); % index of predicted char

        xt = zeros(K,1);
        xt(ii) = 1;
        xiis(t) = ii;
        Y(ii, t) = 1;
    end
end

function [loss, hs, as, P] = ForwardPass(h0, RNN, X, Y)
    ht = h0; % size mx1
    n = size(X, 2);
    m = size(RNN.b, 1);
    K = size(Y, 1);
    
    hs = zeros(m,n+1);
    hs(:,1) = h0;
    as = zeros(m,n);
    loss = 0;
    P = zeros(K, n);
    for t = 1:n
        at = RNN.W * ht + RNN.U * X(:,t) + RNN.b;
        ht = tanh(at);
        ot = RNN.V * ht + RNN.c;
        pt = SoftMax(ot);
        
        hs(:,t+1) = ht;
        as(:,t) = at;
        loss = loss - transpose(Y(:,t)) * log(pt);
        P(:,t) = pt;
    end
end

function grads = ComputeGradsAna(hs, as, Y, X, P, RNN)
    n = size(hs, 2) - 1;
    grad_o = - transpose(Y - P); 
    grad_h = CalculateGradH(hs, grad_o, RNN, as);
    grad_a = - grad_h .* transpose(1 - tanh(as) .* tanh(as));
    grad_W = - transpose(grad_a) * transpose(hs(:,1:n));
    grad_V =  transpose(grad_o) * transpose(hs(:, 2:n+1)); 
    grad_U = - transpose(grad_a) * transpose(X);
    grad_c = transpose(sum(grad_o, 1)); 
    grad_b = transpose(- sum(grad_a, 1));
    grads = struct('b', grad_b, 'c', grad_c, 'U', grad_U, 'W', grad_W, 'V', grad_V);
    % clip gradients (remove before testing gradients)
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
end

function grad_h = CalculateGradH(hs, grad_o, RNN, as)
    grad_h = zeros(size(grad_o, 1), size(hs,1));
    T = size(grad_h, 1);
    grad_h(T,:) = grad_o(T,:) * RNN.V;
    for t = T-1:-1:1
        grad_h(t,:) = grad_o(t,:) * RNN.V + grad_h(t+1,:) .* transpose(1 - tanh(as(t+1)) .* tanh(as(t+1))) * RNN.W;
    end
end

% test gradients
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
X = charsToOneHot(X_chars, K, char_to_ind); 
Y = charsToOneHot(Y_chars, K, char_to_ind);
h0 = zeros(m,1);
% test gradients
[loss, hs, as, P] = ForwardPass(h0, RNN, X, Y);
grads_ana = ComputeGradsAna(hs, as, Y, X, P, RNN);
grads_num = ComputeGradsNum(X, Y, RNN, 1e-4);
for f = fieldnames(grads_ana)'
    max_abs_diff.(f{1}) = max(max(abs(grads_ana.(f{1}) - grads_num.(f{1}))));
    max_rel_diff.(f{1}) = max(max(abs(ComputeRelativeError(grads_ana.(f{1}), grads_num.(f{1})))));
end

function err = ComputeRelativeError(gn, ga)
    err = abs(ga - gn) ./ max(1e-4, abs(ga) + abs(gn));
end

function [new_RNN, smooth_losses] = TrainNetwork(book_data, nr_iterations, seq_length, RNN, char_to_ind, eta, ind_to_char)
    m = size(RNN.b, 1);
    K = size(RNN.c, 1);
    smooth_losses = zeros(nr_iterations, 1);
    smooth_loss = 0;
    G = struct('b', zeros(m,1), 'c', zeros(K,1), 'U', zeros(m, K), 'W', zeros(m, m), 'V', zeros(K, m));
    e = 1;
    hprev = zeros(m,1);
    for i = 1:nr_iterations
        X_chars = book_data(e:e+seq_length-1);
        Y_chars = book_data(e+1:e+seq_length);
        X = charsToOneHot(X_chars, K, char_to_ind);
        Y = charsToOneHot(Y_chars, K, char_to_ind);
        [loss, hs, as, P] = ForwardPass(hprev, RNN, X, Y);
        if i==1
            smooth_loss = loss;
        end
        smooth_loss = .999* smooth_loss + .001 * loss;
        grads = ComputeGradsAna(hs, as, Y, X, P, RNN);
        [RNN, G] = AdaGradUpdateStep(RNN, grads, eta, G);
        hprev = hs(:, size(hs, 2)); % last computed hidden state
        smooth_losses(i) = smooth_loss;

        if(mod(i,10000) == 0)
            % disp(['iter = ', num2str(i), ', loss = ', num2str(smooth_loss)]);
            % disp(['iteration ', num2str(i), ': ', oneHotToChars(SynthesizeText(RNN, zeros(m,1), ' ', 200, char_to_ind), ind_to_char)]);
        end
        e = e+seq_length;
        if e + seq_length > length(book_data)
            e = 1;
            hprev = zeros(m,1);
        end
    end
    new_RNN = RNN;
end

function [new_RNN, G] = AdaGradUpdateStep(RNN, grads, mu, G)
    for f = fieldnames(RNN)'
        G.(f{1}) = G.(f{1}) + grads.(f{1}) .* grads.(f{1});
        new_RNN.(f{1}) = RNN.(f{1}) - (mu./sqrt(G.(f{1}) + eps)) .* grads.(f{1});
    end
end

nr_iterations = round( 7 * (length(book_data) ./ seq_length));
[new_RNN, smooth_losses] = TrainNetwork(book_data, nr_iterations, seq_length, RNN, char_to_ind, eta, ind_to_char);

disp(oneHotToChars(SynthesizeText(new_RNN, zeros(m,1), ' ', 1000, char_to_ind), ind_to_char));

PlotLoss(smooth_losses);

function PlotLoss(smooth_losses)
    figure();
    plot(1:length(smooth_losses), smooth_losses);
    title('smooth loss function');
    xlabel('iteration');
    ylabel('smooth loss');
end
