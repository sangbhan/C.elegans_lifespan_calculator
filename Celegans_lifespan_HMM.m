% This program predicts individual's lifespan using early adulthood health data.

formatSpec = '%f';
filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_train.txt';
fileID = fopen(filename,'r');
shortlived_train = fscanf(fileID, formatSpec);
fclose(fileID);

filename = 'C:\Users\Sangbin\Desktop\Data\normal_lived_train.txt';
fileID = fopen(filename,'r');
normallived_train = fscanf(fileID, formatSpec);
fclose(fileID);

filename = 'C:\Users\Sangbin\Desktop\Data\long_lived_train.txt';
fileID = fopen(filename,'r');
longlived_train = fscanf(fileID, formatSpec);
fclose(fileID);

STATE_NUM = 1:30;
EMISSION_NUM = 27;

maxiter = 10000;
tol = 1e-4;

filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_test_accuracy.txt';
shortlived_test_a = table2array(readtable(filename, 'Format', formatSpec));
shortlived_test_accuracy = 1:30;

filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_train_accuracy.txt';
shortlived_train_a = table2array(readtable(filename,'Format', formatSpec));
shortlived_train_accuracy = 1:30;

filename = 'C:\Users\Sangbin\Desktop\Data\normal_lived_train_accuracy.txt';
normallived_train_a = table2array(readtable(filename,'Format', formatSpec));
normallived_train_accuracy = 1:30;

filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_test_accuracy.txt';
normallived_test_a = table2array(readtable(filename, 'Format', formatSpec));
normallived_test_accuracy = 1:30;
        
filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_train_accuracy.txt';
longlived_train_a = table2array(readtable(filename,'Format', formatSpec));
longlived_train_accuracy = 1:30;

filename = 'C:\Users\Sangbin\Desktop\Data\short_lived_test_accuracy.txt';
longlived_test_a = table2array(readtable(filename, 'Format', formatSpec));
longlived_test_accuracy = 1:30;

for j = 1:30

    TRANS_INIT = (ones(STATE_NUM(j), STATE_NUM(j)) + eye(STATE_NUM(j), STATE_NUM(j)))/(STATE_NUM(j) + 1);
    EMIS_INIT = ones(STATE_NUM(j), EMISSION_NUM)/EMISSION_NUM;

    [TRANS_short, EMIS_short] = hmmtrain(shortlived_train, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    [TRANS_normal, EMIS_normal] = hmmtrain(normallived_train, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    [TRANS_long, EMIS_long] = hmmtrain(longlived_train, TRANS_INIT, EMIS_INIT, 'tolerance', tol, 'maxiterations', maxiter);
    
    formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';

    for i = 1:size(shortlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            shortlived_train_accuracy(j) = shortlived_train_accuracy(j) + 1;

        end

    end

    shortlived_train_accuracy(j) = shortlived_train_accuracy(j) / size(shortlived_train_a, 1);

    for i = 1:size(shortlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(shortlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(shortlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(shortlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            shortlived_test_accuracy(j) = shortlived_test_accuracy(j) + 1;

        end

    end

    shortlived_test_accuracy(j) = shortlived_test_accuracy(j) / size(shortlived_test_a, 1);

    for i = 1:size(normallived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            normallived_train_accuracy(j) = normallived_train_accuracy(j) + 1;

        end

    end

    normallived_train_accuracy(j) = normallived_train_accuracy(j) / size(normallived_train_a, 1);

    for i = 1:size(normallived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(normallived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(normallived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(normallived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            normallived_test_accuracy(j) = normallived_test_accuracy(j) + 1;

        end

    end

    normallived_test_accuracy(j) = normallived_test_accuracy(j) / size(normallived_test_a, 1);

    for i = 1:size(longlived_train_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_train_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_train_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_train_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            longlived_train_accuracy(j) = longlived_train_accuracy(j) + 1;

        end

    end

    longlived_train_accuracy(j) = longlived_train_accuracy(j) / size(longlived_train_a, 1);

    for i = 1:size(longlived_test_a, 1)

        logpseq = [0 0 0];
        [PSTATES, logpseq(1)] = hmmdecode(longlived_test_a(i, :), TRANS_short, EMIS_short);
        [PSTATES, logpseq(2)] = hmmdecode(longlived_test_a(i, :), TRANS_normal, EMIS_normal);
        [PSTATES, logpseq(3)] = hmmdecode(longlived_test_a(i, :), TRANS_long, EMIS_long);
        [M, A] = max(logpseq);

        if A ~= 1

            longlived_test_accuracy(j) = longlived_test_accuracy(j) + 1;

        end

    end

    longlived_test_accuracy(j) = longlived_test_accuracy(j) / size(longlived_test_a, 1);

end

plot(1:30, shortlived_train_accuracy, '-v', 1:30, shortlived_test_accuracy, '->', 1:30, normallived_train_accuracy, '-<', 1:30, normallived_test_accuracy, '-s', 1:30, longlived_train_accuracy, '-^', 1:30, longlived_test_accuracy, '-o')
legend('short-lived training accuracy', 'short-lived test accuracy', 'short-lived training accuracy', 'normal-lived test accuracy', 'long-lived training accuracy', 'long-lived test accuracy')
