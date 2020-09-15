import fasttext

if __name__ == "__main__":
    model = fasttext.train_unsupervised("data/item_sequence_tiny.csv", "data/item_meta.csv", 4)
    # __contains__ が使える
    print(f"the model has 52552294-20: {'52552294-20' in model}")
    print(f"the model has 50436994-16_: {'50436994-16_' in model}")
    print(model.get_nearest_neighbors("52552294-20", k=10))
    # 未知語も探せるが、例外で良い
    print(model.get_nearest_neighbors("50436994-16_", k=10))
    # sideinfo で探す場合
    print(model.get_nearest_neighbors("20\t2028\t12252\t481", k=10))
    model.save_model('model.bin')
    m = fasttext.load_model('model.bin')
    print(m.get_nearest_neighbors("20\t2028\t12252\t481", k=10))
