import os
from os.path import join
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    
    if getattr(args, "save_features", False):
        np.save(join(args.save_dir, f"features_{test_method}.npy"),
                all_features.astype("float16"))
        logging.info(f"Saved descriptors to {join(args.save_dir, f'features_{test_method}.npy')}")

    whiten_blocks = getattr(args, "whiten_blocks", "")
    if whiten_blocks:
        # Multi-view calibration: each token family (semantic cluster tokens,
        # multi-scale GeM cells, ...) gets its own PCA-whitening, fitted on the
        # database, before the calibrated blocks are concatenated.
        from sklearn.decomposition import PCA
        tokens = [int(t) for t in whiten_blocks.split(",")]
        dims_arg = getattr(args, "whiten_block_dims", "")
        dims = [int(d) for d in dims_arg.split(",")] if dims_arg else \
               [getattr(args, "whiten_dim", 256)] * len(tokens)
        assert len(dims) == len(tokens)
        w_arg = getattr(args, "whiten_block_weights", "auto")
        if w_arg == "auto":
            # View reliability scales with the size of its calibrated subspace:
            # a view that needed k_v whitened dimensions carries k_v units of
            # information, and per-view L2 normalisation throws that away.
            weights = [float(np.sqrt(d)) for d in dims]
        elif w_arg == "equal":
            weights = [1.0] * len(dims)
        else:
            weights = [float(w) for w in w_arg.split(",")]
        assert len(weights) == len(tokens)
        wmax = max(weights)
        weights = [w / wmax for w in weights]
        D = all_features.shape[1] // sum(tokens)
        assert D * sum(tokens) == all_features.shape[1], (all_features.shape, tokens)
        fit_n = eval_ds.database_num if getattr(args, "whiten_fit", "db") == "db" else len(all_features)
        blocks, offset = [], 0
        for n_tok, d, w in zip(tokens, dims, weights):
            sl = slice(offset, offset + n_tok * D)
            offset += n_tok * D
            pca = PCA(n_components=min(d, n_tok * D, fit_n - 1), whiten=True,
                      svd_solver="randomized", random_state=args.seed)
            pca.fit(all_features[:fit_n, sl])
            z = pca.transform(all_features[:, sl]).astype("float32")
            z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
            blocks.append(z * w)
            logging.info(f"Calibrated block {n_tok} tokens -> {z.shape[1]} dims, weight {w:.3f}")
        all_features = np.concatenate(blocks, axis=1)
        all_features /= (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)
        args.features_dim = all_features.shape[1]

    whiten_dim = getattr(args, "whiten_dim", 0)
    if whiten_dim and not whiten_blocks:
        # Training-free, label-free descriptor calibration: PCA-whitening fitted
        # on the database (or on db+queries) removes the dominant anisotropic
        # directions shared by all descriptors of a frozen backbone.
        from sklearn.decomposition import PCA
        fit_n = eval_ds.database_num if getattr(args, "whiten_fit", "db") == "db" else len(all_features)
        pca = PCA(n_components=whiten_dim, whiten=True, svd_solver="randomized",
                  random_state=args.seed)
        pca.fit(all_features[:fit_n])
        all_features = pca.transform(all_features).astype("float32")
        all_features /= (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)
        args.features_dim = whiten_dim
        logging.info(f"Whitened descriptors to {whiten_dim} dims (fit on {fit_n} images)")

    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    
    # ... (接在 recalls 计算代码之后) ...
    import json
    import os
    
    # 1. 定义保存路径
    # 建议用 args.resume (模型路径) 或 args.save_dir 来区分文件名
    # 例如: results_hard_resize_v1.json
    result_filename = f"results_{args.test_method}_{getattr(args, 'experiment_name', 'default')}.json"
    result_path = os.path.join(args.save_dir, result_filename)
    
    logging.info(f"Saving detailed results to {result_path} ...")
    
    detailed_results = {}
    
    # 2. 遍历收集数据
    for q_idx, preds in enumerate(predictions):
        positives = positives_per_query[q_idx]
        top1_pred = preds[0] # 取 Top-1 预测
        is_correct = bool(top1_pred in positives) # 必须要转成 bool 才能存 JSON
        
        # 获取路径（请根据你的 Dataset 类修改属性名）
        # 假设 eval_ds.queries_paths 存的是 query 路径列表
        # 假设 eval_ds.database_paths 存的是 database 路径列表
        q_path_full = eval_ds.queries_paths[q_idx]
        p_path_full = eval_ds.database_paths[top1_pred]
        
        # 为了方便阅读，只存文件名 (basename)
        q_name = os.path.basename(q_path_full)
        
        detailed_results[q_name] = {
            "q_idx": q_idx,
            "is_correct": is_correct,
            "pred_db_idx": int(top1_pred),
            "q_path": q_path_full,  # 存完整路径方便后续可视化读取
            "p_path": p_path_full   # 存检索到的结果路径
        }
        
    # 3. 写入文件
    with open(result_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    logging.info("Results saved.")
    
    return recalls, recalls_str