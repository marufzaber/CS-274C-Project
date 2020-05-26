from project.train import cnn

if __name__ == '__main__':
    cnn.train(70, 100, 0.001, 'model_output')
