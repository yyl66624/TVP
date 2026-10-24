import os
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #------------------------------------------      core set          --------------------------------------------------
    parser.add_argument('--num_clusters', type=int, default=16, help="Number of clusters for clustering layer.")
    parser.add_argument("--foundation_model_path", type=str, 
                        default="/data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth",
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                "cct384", "vitb16_224", "vitb16_384", "dinov2", "dinov2_l",
                                "dinov2_g", "swin"])
    parser.add_argument('--resize', type=int, default=[336, 336], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--eval_dataset_name", type=str, 
                        default="msls", 
                        choices=["msls", "nordland", "pitts30k", "pitts250k", "sped",
                                 "san_francisco", "st_lucia", "tokyo247","amstertime", "eynsham",
                                 "svox", "svox_night", "svox_overcast", "svox_rain", "svox_snow", "svox_sun",
                                 "freiburg_12w", "freiburg_15w", "itlp_day_night_0506",
                                 "itlp_day_night_0708", "itlp_season_0005", "itlp_indoor",
                                 "robotcar_sun_night", "robotcar_snow", "robotcar_day_night",
                                 "covpr_indoor", "nyc_indoor", "etis_stmartin", "motive"])
    parser.add_argument("--eval_datasets_folder", type=str, default='/data/datasets/vpr', help="Path with all datasets")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--infer_batch_size", type=int, default=50, help="Batch size for inference (caching and testing)")
    # ---- NLPrompt-inspired training-free aggregation (OT-CAM) ----
    parser.add_argument("--agg_mode", type=str, default="tf", choices=["tf", "st", "ot"],
                        help="Cluster aggregation: tf = TF_CAM (softmax over clusters), "
                             "st = softmax over tokens, ot = optimal transport")
    parser.add_argument("--st_temp", type=float, default=1.0,
                        help="Temperature for token-softmax aggregation")
    parser.add_argument("--tf_temp", type=float, default=1.0,
                        help="Temperature for the TF_CAM cluster softmax")
    parser.add_argument("--tf_variant", type=str, default="n_minus_1_cls",
                        choices=["n_minus_1_cls", "n_cls", "n_no_cls"],
                        help="TF_VPR token layout: (n-1 clusters + CLS) | (n clusters + CLS) | (n clusters)")
    parser.add_argument("--tf_global_token", type=str, default="cls",
                        choices=["cls", "gem", "avg", "both", "none"],
                        help="Global token paired with the TF cluster tokens: CLS | GeM | "
                             "mean(CLS,GeM) | both CLS and GeM | none")
    parser.add_argument("--tf_fusion_gem", action="store_true",
                        help="Append the 13 multi-scale GeM tokens (2x2 + 3x3) to the TF descriptor")
    parser.add_argument("--tf_mean_view", action="store_true",
                        help="Append a globally pooled (GeM) token of the raw patch tokens")
    parser.add_argument("--ot_reg", type=float, default=0.2, help="Sinkhorn entropy regularisation")
    parser.add_argument("--ot_iters", type=int, default=30, help="Sinkhorn iterations")
    parser.add_argument("--ot_temp", type=float, default=1.0, help="Logit temperature before OT")
    parser.add_argument("--ot_dustbin", type=float, default=0.0,
                        help="Mass assigned to the outlier bin (0 disables purification)")
    parser.add_argument("--ot_dustbin_cost", type=float, default=0.0,
                        help="Cost of the outlier bin (higher = fewer tokens dropped)")
    parser.add_argument("--ot_cost_clip", type=float, default=None,
                        help="If set, bound logits with c*tanh(x/c) (robust influence)")
    parser.add_argument("--cluster_cache", type=str, default=None,
                        help="Path to save/load the k-means prototypes (shared across ablations)")
    parser.add_argument("--gam_k", type=int, default=8, help="TF-GAM neighbours")
    parser.add_argument("--gam_lambda", type=float, default=0.8, help="TF-GAM self weight")
    parser.add_argument("--skip_gam", action="store_true", help="Disable TF-GAM (ablation)")
    parser.add_argument("--cam_norm_first", action="store_true",
                        help="Normalise tokens before computing cluster logits (cosine semantics)")
    parser.add_argument("--kmeans_descs", type=int, default=None,
                        help="Descriptors sampled per image for k-means (default: upstream heuristic)")
    parser.add_argument("--kmeans_images", type=int, default=0,
                        help="Cap the number of test images used to fit the k-means prototypes "
                             "(0 = all, upstream). Useful for very large test sets.")
    parser.add_argument("--kmeans_on_gam", action="store_true",
                        help="Run k-means on TF-GAM refined tokens instead of raw tokens")
    parser.add_argument("--lang_prompt_dir", type=str, default=None,
                        help="Local CLIP model dir enabling the natural-language prompt bank")
    parser.add_argument("--lang_gamma", type=float, default=0.3,
                        help="Strength of prompt-guided token reweighting")
    parser.add_argument("--lang_tokens", type=int, default=0,
                        help="Number of language-anchored tokens appended to the descriptor (0 disables)")
    parser.add_argument("--lang_temp", type=float, default=0.1,
                        help="Temperature of the prompt-to-token attention")
    parser.add_argument("--mode", type=str, default="TF_VPR",
                        choices=["TF_VPR", "GeM_CLS", "CLS", "Mean", "GeM_Mean",
                                 "CLS_Mean", "VLAD", "TF_VLAD"],
                        help="Mode choice for inference (caching and testing)")
    parser.add_argument("--seed", type=int, default=1)
    #---------------------------------------------------------------------------------------------------------------------
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=float, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="If set, apply PCA to the global descriptors")
    parser.add_argument("--fc_output_dim", type=int, default=None,
                        help="Optional FC output dimension (unused for the training-free modes)")
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="Tag used in the exported per-query result filename")
    parser.add_argument("--save_features", action="store_true",
                        help="Dump the test-set descriptors to the run folder (for offline post-processing)")
    parser.add_argument("--whiten_dim", type=int, default=0,
                        help="If >0, PCA-whiten the descriptors to this dimension before retrieval")
    parser.add_argument("--whiten_fit", type=str, default="all", choices=["db", "all"],
                        help="Fit the whitening on the database only or on db+queries")
    parser.add_argument("--whiten_blocks", type=str, default="",
                        help="Comma-separated token counts per descriptor block, e.g. "
                             "'17,13' for (CLS+16 cluster tokens) + (13 GeM cells); "
                             "each block is PCA-whitened independently")
    parser.add_argument("--whiten_block_dims", type=str, default="",
                        help="Comma-separated whitening dims per block, e.g. '2048,1024'")
    parser.add_argument("--whiten_block_weights", type=str, default="auto",
                        help="View weights after per-block calibration: 'auto' (sqrt of the "
                             "block dimension, normalised by the largest), 'equal', or a "
                             "comma-separated list, e.g. '1,0.7,0.6'")
    parser.add_argument("--vlad_k", type=int, default=64,
                        help="Number of VLAD centres for --mode VLAD")
    parser.add_argument("--vlad_centers", type=str, default="",
                        help="Path to the VLAD vocabulary (.npy, K x D, L2-normalised rows)")
    parser.add_argument("--vlad_centers2", type=str, default="",
                        help="Second VLAD vocabulary for --mode TF_VLAD "
                             "(K x D .npy); its block is appended after the first")
    parser.add_argument("--vlad_soft", type=float, default=0.0,
                        help="Soft-assignment temperature for --mode VLAD "
                             "(0 = hard assignment)")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./test/)")
    args = parser.parse_args()

    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --eval_datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable, e.g.\n" +
                            "export DATASETS_FOLDER=/path/to/datasets")

    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")

    return args

