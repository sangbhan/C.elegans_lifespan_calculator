% This program predicts C.elegans lifespan using early adulthood health data with a hidden Markov model.

formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';

% open short-lived test data for accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_test_accuracy.txt';
shortlived_test_a = table2array(readtable(filename, 'Format', formatSpec));
shortlived_test_accuracy = zeros(1, 7);

% open short-lived training data for training and accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_train_accuracy.txt';
shortlived_train_a = table2array(readtable(filename,'Format', formatSpec));
shortlived_train_accuracy = zeros(1, 7);

% open normal-lived test data for accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\normal_lived_test_accuracy.txt';
normallived_test_a = table2array(readtable(filename, 'Format', formatSpec));
normallived_test_accuracy = zeros(1, 7);

% open normal-lived training data for training and accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\normal_lived_train_accuracy.txt';
normallived_train_a = table2array(readtable(filename,'Format', formatSpec));
normallived_train_accuracy = zeros(1, 7);

% open long-lived test data for accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\long_lived_test_accuracy.txt';
longlived_test_a = table2array(readtable(filename, 'Format', formatSpec));
longlived_test_accuracy = zeros(1, 7);

% open long-lived training data for training and accuracy calculation
filename = 'C:\Users\Sangbin\Desktop\Data\long_lived_train_accuracy.txt';
longlived_train_a = table2array(readtable(filename,'Format', formatSpec));
longlived_train_accuracy = zeros(1, 7);

STATE_NUM = 1:3:19;
EMISSION_NUM = 27;

maxiter = 100;
tol = 1e-6;

% calculate accuracy by number of hidden states
for j = 1:7
    
    % initialize transition matrix and emission matrix
    TRANS_INIT = (ones(STATE_NUM(j), STATE_NUM(j)) + eye(STATE_NUM(j))) / (STATE_NUM(j) + 1);
    EMIS_INIT = ones(STATE_NUM(j), EMISSION_NUM) / EMISSION_NUM;
    
    % train transition matrix and emission matrix
    [TRANS_short, EMIS_short] = hmmtrain(shortlived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    [TRANS_normal, EMIS_normal] = hmmtrain(normallived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    [TRANS_long, EMIS_long] = hmmtrain(longlived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    
    % calculate short-lived training accuracy by number of hidden states
    for i = 1:size(shortlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 1

            shortlived_train_accuracy(j) = shortlived_train_accuracy(j) + 1;

        end

    end

    shortlived_train_accuracy(j) = shortlived_train_accuracy(j) * 100 / size(shortlived_train_a, 1);

    % calculate short-lived test accuracy by number of hidden states
    for i = 1:size(shortlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 1

            shortlived_test_accuracy(j) = shortlived_test_accuracy(j) + 1;

        end

    end

    shortlived_test_accuracy(j) = shortlived_test_accuracy(j) * 100 / size(shortlived_test_a, 1);

    % calculate normal-lived training accuracy by number of hidden states
    for i = 1:size(normallived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 2

            normallived_train_accuracy(j) = normallived_train_accuracy(j) + 1;

        end

    end

    normallived_train_accuracy(j) = normallived_train_accuracy(j) * 100 / size(normallived_train_a, 1);

    % calculate normal-lived test accuracy by number of hidden states
    for i = 1:size(normallived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 2

            normallived_test_accuracy(j) = normallived_test_accuracy(j) + 1;

        end

    end

    normallived_test_accuracy(j) = normallived_test_accuracy(j) * 100 / size(normallived_test_a, 1);

    % calculate long-lived training accuracy by number of hidden states
    for i = 1:size(longlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 3

            longlived_train_accuracy(j) = longlived_train_accuracy(j) + 1;

        end

    end

    longlived_train_accuracy(j) = longlived_train_accuracy(j) * 100 / size(longlived_train_a, 1);

    % calculate long-lived test accuracy by number of hidden states
    for i = 1:size(longlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 3

            longlived_test_accuracy(j) = longlived_test_accuracy(j) + 1;

        end

    end
    
    longlived_test_accuracy(j) = longlived_test_accuracy(j) * 100 / size(longlived_test_a, 1);

end

% plot accuracy by number of hidden states
plot(1:3:19, shortlived_train_accuracy, '-v', 1:3:19, shortlived_test_accuracy, '->', 1:3:19, normallived_train_accuracy, '-<', 1:3:19, normallived_test_accuracy, '-s', 1:3:19, longlived_train_accuracy, '-^', 1:3:19, longlived_test_accuracy, '-o')
axis([0 20 0 70])
legend('short-lived training accuracy', 'short-lived test accuracy', 'normal-lived training accuracy', 'normal-lived test accuracy', 'long-lived training accuracy', 'long-lived test accuracy')
xlabel('Number of hidden states')
ylabel('Accuracy (%)')
title('Accuracy by number of hidden states', 'FontSize', 12)



% initialize transition matrix and emission matrix
STATE_NUM = 4;
TRANS_INIT = (ones(STATE_NUM, STATE_NUM) + eye(STATE_NUM)) / (STATE_NUM + 1);
EMIS_INIT = ones(STATE_NUM, EMISSION_NUM) / EMISSION_NUM;

maxiter = 1000;
tol = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7];

shortlived_test_accuracy = zeros(1, 7);
shortlived_train_accuracy = zeros(1, 7);
normallived_test_accuracy = zeros(1, 7);
normallived_train_accuracy = zeros(1, 7);
longlived_test_accuracy = zeros(1, 7);
longlived_train_accuracy = zeros(1, 7);

% calculate accuracy by tolerance
for j = 1:7
    
    % train transition matrix and emission matrix
    [TRANS_short, EMIS_short] = hmmtrain(shortlived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol(j), 'maxiterations', maxiter);
    [TRANS_normal, EMIS_normal] = hmmtrain(normallived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol(j), 'maxiterations', maxiter);
    [TRANS_long, EMIS_long] = hmmtrain(longlived_train_a, TRANS_INIT, EMIS_INIT, 'tolerance', tol(j), 'maxiterations', maxiter);
    
    % calculate short-lived training accuracy by tolerance
    for i = 1:size(shortlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 1

            shortlived_train_accuracy(j) = shortlived_train_accuracy(j) + 1;

        end

    end

    shortlived_train_accuracy(j) = shortlived_train_accuracy(j) * 100 / size(shortlived_train_a, 1);

    % calculate short-lived test accuracy by tolerance
    for i = 1:size(shortlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 1

            shortlived_test_accuracy(j) = shortlived_test_accuracy(j) + 1;

        end

    end

    shortlived_test_accuracy(j) = shortlived_test_accuracy(j) * 100 / size(shortlived_test_a, 1);

    % calculate normal-lived training accuracy by tolerance
    for i = 1:size(normallived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 2

            normallived_train_accuracy(j) = normallived_train_accuracy(j) + 1;

        end

    end

    normallived_train_accuracy(j) = normallived_train_accuracy(j) * 100 / size(normallived_train_a, 1);

    % calculate normal-lived test accuracy by tolerance
    for i = 1:size(normallived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 2

            normallived_test_accuracy(j) = normallived_test_accuracy(j) + 1;

        end

    end

    normallived_test_accuracy(j) = normallived_test_accuracy(j) * 100 / size(normallived_test_a, 1);

    % calculate long-lived training accuracy by tolerance
    for i = 1:size(longlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 3

            longlived_train_accuracy(j) = longlived_train_accuracy(j) + 1;

        end

    end

    longlived_train_accuracy(j) = longlived_train_accuracy(j) * 100 / size(longlived_train_a, 1);

    % calculate long-lived test accuracy by tolerance
    for i = 1:size(longlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A == 3

            longlived_test_accuracy(j) = longlived_test_accuracy(j) + 1;

        end

    end
    
    longlived_test_accuracy(j) = longlived_test_accuracy(j) * 100 / size(longlived_test_a, 1);

end

% plot accuracy by tolerance
plot(1:7, shortlived_train_accuracy, '-v', 1:7, shortlived_test_accuracy, '->', 1:7, normallived_train_accuracy, '-<', 1:7, normallived_test_accuracy, '-s', 1:7, longlived_train_accuracy, '-^', 1:7, longlived_test_accuracy, '-o')
axis([0 8 0 70])
legend('short-lived training accuracy', 'short-lived test accuracy', 'normal-lived training accuracy', 'normal-lived test accuracy', 'long-lived training accuracy', 'long-lived test accuracy')
xlabel('-log(tolerance)')
ylabel('Accuracy (%)')
title('Accuracy by -log(tolerance)', 'FontSize', 12)
