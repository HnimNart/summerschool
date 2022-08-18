import os.path

import numpy as np
import pandas

from fungi_classification import *

def evaluate_uncertainty(team, team_pw, im_dir, nw_dir,
                         best_model="DF20M-EfficientNet-B0_best_accuracy.pth",
                         use_set='train_set',
                         imageNet=False):
    """
        Evaluate trained network on the test set and submit the results to the challenge database.
        The scores can be extracted using compute_challenge_score.
        The function can also be used to evaluate on the final set
    """
    # Use 'test-set' for the set of data that can evaluated several times
    # Use 'final-set' for the final set that will be used in the final score of the challenge


    print(f"Evaluating on {use_set}")

    if use_set == 'labeled_train_set':

        nw_dir = 'FungiNetwork'
        data_file = os.path.join(nw_dir, "data_with_labels.csv")
        df = pd.read_csv(data_file)

    else:
        imgs_and_data = fcp.get_data_set(team, team_pw, use_set)
        df = pd.DataFrame(imgs_and_data, columns=['image', 'class'])
        df['image'] = df.apply(
            lambda x: im_dir + x['image'] + '.JPG', axis=1)

    test_dataset = NetworkFungiDataset(df, transform=get_transforms(data='valid'))

    batch_sz = 32
    n_workers = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if not imageNet:

        n_classes = 183
        best_trained_model = os.path.join(nw_dir, best_model)
        log_file = os.path.join(nw_dir, "FungiEvaluation.log")
        data_stats_file = os.path.join(nw_dir, "fungi_class_stats.csv")

        logger = init_logger(log_file)
        model = EfficientNet.from_name('efficientnet-b0', num_classes=n_classes)
        checkpoint = torch.load(best_trained_model)
        model.load_state_dict(checkpoint)

    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    model.to(device)

    model.eval()
    preds = np.zeros((len(test_dataset)))
    entropy = np.zeros((len(test_dataset)))

    # preds_raw = []

    names = list(df['image'])
    names = np.array([os.path.basename(i) for i in names])


    for i, (images, labels) in tqdm.tqdm(enumerate(test_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)
        y_preds = y_preds.to('cpu').numpy()

        y_entropy = y_preds
        y_entropy = np.exp(y_entropy) / np.sum(np.exp(y_entropy), axis=-1, keepdims=True)


        y_entropy = - np.sum(y_entropy * np.log(y_entropy + 1e-9), axis=1)

        entropy[i * batch_sz: (i + 1) * batch_sz] = y_entropy

    #
    res = pandas.DataFrame({'name': names,
                            'entropy': entropy})

    prefix = 'trained' if not imageNet else 'imageNet'

    res.to_csv(f'{prefix}_{use_set}_entropy.csv')

if __name__ == '__main__':


    team = "CuriousTermite"
    team_pw = "fungi87"

    # where is the full set of images placed
    image_dir = "./../data/DF20M/"

    # where should log files, temporary files and trained models be placed
    network_dir = "./../log/"

    get_participant_credits(team, team_pw)
    print_data_set_numbers(team, team_pw)
    # request_random_labels(team, team_pw)
    get_all_data_with_labels(team, team_pw, image_dir, network_dir)

    use_set = 'test_set'
    # use_set = 'final_set'
    use_set = 'train_set'
    # use_set = 'labeled_train_set'

    evaluate_uncertainty(
        team, team_pw,
        image_dir, network_dir,
         best_model="DF20M-EfficientNet-B0_best_accuracy.pth",
         imageNet=False,
         use_set=use_set
         )


