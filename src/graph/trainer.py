import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold

from graph import GraphDataset
from utils import train, validate, test_model, collate, save_checkpoint, setup_environment, setup_model, load_data

def main(args):
    """Main function"""
    args.result_dir = setup_environment(args)
    (smiles_train, labels_train), (smiles_test, labels_test) = load_data(args)
    
    tqdm.write(f"Data loaded: {np.ceil(100 * len(smiles_train) / (len(smiles_train) + len(smiles_test)))} % train set, "
               f"{np.floor(100 * len(smiles_test) / (len(smiles_train) + len(smiles_test)))} % test set")
    
    if not args.skip_cv:
        tqdm.write(f"Cross validation with {args.cv} folds")
        kf = KFold(n_splits=args.cv, random_state=args.seed, shuffle=True)
        
        for cv_index, (train_indices, valid_indices) in enumerate(kf.split(range(len(smiles_train)))):
            fname = f"cvid{cv_index}"
            model, model_arch = setup_model(args, saliency=False)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            
            train_full_dataset = GraphDataset(smiles_train, labels_train, add_descriptor=args.add_features, extra_in_dim=args.num_feat)
            train_loader = DataLoader(train_full_dataset, batch_size=args.batch_size, 
                                      sampler=SubsetRandomSampler(train_indices), collate_fn=collate)
            val_loader = DataLoader(train_full_dataset, batch_size=args.batch_size, 
                                    sampler=SubsetRandomSampler(valid_indices), collate_fn=collate)
            
            # training
            if args.train:
                best_rmse = float('inf')
                tqdm.write(f"Training...CV: {cv_index}")
                for epoch in range(args.start_epoch, args.epochs):
                    train_loss, train_rmse = train(train_loader, model, loss_fn, optimizer, args)
                    val_loss, val_rmse = validate(val_loader, model, args)
                    is_best = val_rmse < best_rmse
                    best_rmse = min(val_rmse, best_rmse)
                    if args.print_result:
                        tqdm.write(f"epoch: {epoch+1}, train loss: {train_loss:.3f}, train rmse: {train_rmse:.3f}, "
                               f"val loss: {val_loss:.3f}, val rmse: {val_rmse:.3f}, best rmse: {best_rmse:.3f}, best model: {is_best}")
                    
                    if is_best:
                        save_checkpoint(
                            {
                                'epoch': epoch + 1, 
                                'model_arch': model_arch, 
                                'state_dict': model.state_dict(), 
                                'best_rmse': best_rmse, 
                                'optimizer': optimizer.state_dict()
                            }, 
                            fname, args.result_dir
                        )
            tqdm.write("Testing the model ...")
            args.start_epoch = 0
            # load checkpoint
            checkpoint = torch.load(fr"{args.result_dir}/{fname}.pth.tar")
            best_rmse = checkpoint['best_rmse'] 
            model, model_arch = setup_model(args, saliency=True) 
            model.load_state_dict(checkpoint['state_dict']) 
            tqdm.write(f"=> loaded checkpoint '{fname}' => (epoch: {checkpoint['epoch']}, best rmse: {best_rmse:.3f})")
            # compute prediction 
            test_model(
                graph_it=GraphDataset, 
                model=model, 
                id=fname, 
                train_data=(smiles_train[train_indices], labels_train[train_indices]), 
                val_data=(smiles_train[valid_indices], labels_train[valid_indices]), 
                test_data=(smiles_test, labels_test), 
                args=args
            )
    else:
        tqdm.write("No cross validation")
        train_full_dataset = GraphDataset(smiles=smiles_train, y=labels_train, add_descriptor=args.add_features, extra_in_dim=args.num_feat)
        
        indices = np.arange(len(smiles_train))
        np.random.shuffle(indices)
        split = int(np.floor(0.1 * len(smiles_train)))
        train_indices, valid_indices = indices[split:], indices[:split]
        
        train_loader = DataLoader(train_full_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=collate)
        val_loader = DataLoader(train_full_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(valid_indices), collate_fn=collate)
        
        fname = r"noCV"
        best_rmse = float('inf')
        
        if args.train:
            tqdm.write("Training the model ...")
            model, model_arch = setup_model(args, saliency=False)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            for epoch in range(args.start_epoch, args.epochs):
                train_loss, train_rmse = train(train_loader, model, loss_fn, optimizer, args)
                val_loss, val_rmse = validate(val_loader, model, args)
                is_best = val_rmse < best_rmse
                best_rmse = min(val_rmse, best_rmse)
                if args.print_result:
                    tqdm.write(f"epoch: {epoch+1}, train loss: {train_loss:.3f}, train rmse: {train_rmse:.3f}, "
                           f"val loss: {val_loss:.3f}, val rmse: {val_rmse:.3f}, best rmse: {best_rmse:.3f}, best model: {is_best}")
                
                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_arch': model_arch,
                        'state_dict': model.state_dict(),
                        'best_rmse': best_rmse,
                        'optimizer': optimizer.state_dict(),
                    }, fname, args.result_dir)

        # test
        tqdm.write("Testing the model ...")
        checkpoint = torch.load(fr"{args.result_dir}/{fname}.pth.tar")
        model, model_arch = setup_model(args, saliency=True) 
        model.load_state_dict(checkpoint['state_dict'])
        tqdm.write(f"=> loaded checkpoint '{fname}' (epoch {checkpoint['epoch']}, rmse {checkpoint['best_rmse']})")
        
        test_model(
            graph_it=GraphDataset, 
            model=model, 
            id=fname, 
            train_data=(smiles_train[train_indices], labels_train[train_indices]), 
            val_data=(smiles_train[valid_indices], labels_train[valid_indices]), 
            test_data=(smiles_test, labels_test), 
            args=args
        )
            
    # write args to file
    with open(f'{args.result_dir}/args.txt', 'w') as f:
        f.write(str(args))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN with descriptors')
    parser.add_argument('--parent_directory', type=str, default='/Users/gbemidebe/Documents/GitHub/BatteryInformatics', help='Path to main directory')
    parser.add_argument('--result_directory', default='results/GNN', type=str, help='Path to result directory')
    parser.add_argument('--data_directory', default='data', type=str, help='where the data is stored in parent directory')
    parser.add_argument('--train_data', default='train_data_cleaned.csv', type=str, help='name of train data')
    parser.add_argument('--test_data', default='test_data_cleaned.csv', type=str, help='name of test data')
    parser.add_argument('--add_features', action='store_true', help='if add features')
    parser.add_argument('--skip_cv', action='store_true', help='if skip cross validation')
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=5, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID to use.')
    parser.add_argument('--cv', default=3, type=int, help='k-fold cross validation')
    parser.add_argument('--dim_input', default=74, type=int, help='dimension of input')
    parser.add_argument('--unit_per_layer', default=256, type=int, help='unit per layer')
    parser.add_argument('--seed', default=2020, type=int, help='seed number')
    parser.add_argument('--num_feat', default=6, type=int, help='number of additional features')
    parser.add_argument('--train', action='store_true', help='if train')
    parser.add_argument('--print_result', action='store_true', help='if print result')
    parser.add_argument('--docker', action='store_true', help='if docker')
    args = parser.parse_args()
    main(args)
