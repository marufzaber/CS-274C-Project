from project.train import cnn

if __name__ == '__main__':
    cnn.train(8, 100, 0.0005, 'model_output')
