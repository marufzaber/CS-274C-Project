from project.train import cnn

if __name__ == '__main__':
    cnn.train(32, 100, 0.0005, 'model_output')
