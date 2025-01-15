import numpy as np
import argparse
import yaml
#from
import os


if __name__ == "__main__":
# Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Fairness in IR systems.")

    # add parameters
    parser.add_argument("--task", type=str, choices=["recommendation", "search"], default='recommendation', help='IR tasks')
    parser.add_argument("--stage", type=str, choices=["retrieval", "ranking", "re-ranking"], default="ranking", help="your evaluation stage")
    parser.add_argument("--dataset", type=str, choices=["yelp2018", "Amazon_All_Beauty", "Amazon_Digital_Music", "steam", "AliEC", "clueweb09"], default="steam", help="your dataset")
    parser.add_argument("--train_config_file", type=str, default="train_Ranking.yaml", help="your train yaml file")
    #parser.add_argument("--reprocess", type=str, choices=["yes", "no"], default="no", help="your dataset")
    #parser.add_argument("topk", type=float, default=10, help="ranking size")
    args = parser.parse_args()
    with open(os.path.join(args.task, args.train_config_file), 'r') as f:
        train_config = yaml.safe_load(f)
    train_config['dataset'] = args.dataset
    train_config['stage'] = args.stage
    train_config['task'] = args.task
    print("your training config...")
    print(train_config)
    # parse the args

    print("your args:", args)
    if args.task == "recommendation":
        from recommendation.trainer import RecTrainer
        from recommendation.reranker import RecReRanker
        if args.stage == 'ranking' or args.stage == 'retrieval':
            trainer = RecTrainer(args.dataset, args.stage, train_config)
            trainer.train()
        elif args.stage == 're-ranking':
            reranker = RecReRanker(args.dataset, args.stage, train_config)
            reranker.rerank()
        else:
            raise NotImplementedError("we only support stage in [retrieval, ranking, re-ranking]")
    elif args.task == "search":
        if args.stage == "re-ranking" and args.dataset == "clueweb09":
            from search.trainer import SRDTrainer
            trainer = SRDTrainer(train_config)
            trainer.train()
    else:
        raise NotImplementedError




