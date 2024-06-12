if __name__ == "__main__":
    from src.update import DatasetSplit
    from src.utils import get_dataset
    import numpy as np
    import matplotlib.pyplot as plt

    NUM_CLIENTS, NUM_LABELS = 10, 10
    clients_labels = np.zeros((NUM_CLIENTS, NUM_LABELS))


    class args:
        def __init__(self):
            self.num_channels = 1
            self.iid = 2
            self.num_classes = 10
            self.num_users = 10
            self.dataset = 'mnist'
            self.dirichlet = 0.1


    dataset, _, user_groups = get_dataset(args())
    idxs_users = np.random.choice(range(10), 10, replace=False)
    print(len(idxs_users))
    for idx in idxs_users:
        idxs = user_groups[idx]
        train_dataset = DatasetSplit(dataset, idxs)
        labels = [t[1].item() for t in train_dataset]
        for label in labels:
            clients_labels[idx][label] += 1
        # plt.hist(labels, bins=20, color='skyblue', edgecolor='black')
        # plt.show()
    clients_labels[0]