from project.train import svm

if __name__ == '__main__':
    svm.train(32, 1000, 0.001, 'model_output')
