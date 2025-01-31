import numpy as np
import argparse
import yaml
import os


if __name__ == "__main__":
# Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Fairness in IR systems.")

    # add parameters
    parser.add_argument("--task", type=str, choices=["recommendation", "search"], default='recommendation', help='IR tasks')
    parser.add_argument("--stage", type=str, choices=["pre-processing", "in-processing", "post-processing"],
                        default="post-processing", help="your evaluation stage")
    parser.add_argument("--dataset", type=str, choices=["steam", "clueweb09"], default="steam", help="your dataset")
    parser.add_argument("--train_config_file", type=str, default="train.yaml", help="your train yaml file")
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
        from recommendation.llm_rec import LLMRecommender
        if args.stage == 'in-processing':
            if train_config['use_llm']:
                LLMRecommender = LLMRecommender(args.dataset, args.stage, train_config)
                LLMRecommender.recommend()
            else:
                trainer = RecTrainer(args.dataset, args.stage, train_config)
                trainer.train()
        elif args.stage == 'post-processing':
            reranker = RecReRanker(args.dataset, args.stage, train_config)
            reranker.rerank()
        else:
            raise NotImplementedError("we only support stage in [retrieval, ranking, re-ranking]")
    elif args.task == "search":
        if args.stage == "post-processing":
            if args.dataset != 'clueweb09':
                raise ValueError("For post-processing methods in search, we only support the clueweb09 dataset")
            from search.trainer import SRDTrainer
            trainer = SRDTrainer(train_config)
            trainer.train()
    else:
        raise NotImplementedError




