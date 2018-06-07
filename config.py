# coding:utf-8


class DefaultConfig(object):
    model = 'SuleymanNet'

    root = './data'

    load_model_path = ''
    save_model_path = './model/'

    batch_size = 128
    use_gpu = True
    num_workers = 4

    max_epoch = 10
    lr = 0
    lr_decay = 0.5
    weight_decay = 0

    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == '__main__':
    opt = DefaultConfig()
