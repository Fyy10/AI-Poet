class Config(object):
    num_layers = 3
    data_path = 'tang.npz'
    device = None
    batch_size = 64
    num_workers = 4
    hidden_dim = 512
    embedding_dim = 256
    learning_rate = 1e-3
    epoch = 1
    model_type = 'Seq2Seq'
    model_path = 'CheckPoints/tang_final_{}.pth'.format(model_type)
    model_prefix = 'CheckPoints/tang'
    max_gen_len = 200
    prefix_words = None
