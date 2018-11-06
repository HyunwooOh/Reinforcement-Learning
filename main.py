import argparse
import multiprocessing
import dqn, drqn, a3c
from utils.common_utils import make_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c', type = str)
    ##############################################
    parser.add_argument("--game", default='BreakoutDeterministic', type = str)
    parser.add_argument("--num_frame", default=4, type=int)
    ################  Value Based  ################
    parser.add_argument("--double", default='False', type = str)
    parser.add_argument("--dueling", default='False', type = str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--train_start", default=50000, type = int)
    parser.add_argument("--train_end", default=5000000, type = int)
    parser.add_argument("--target_update_rate", default=10000, type = int)
    parser.add_argument("--memory_size", default=500000, type = int)
    parser.add_argument("--epsilon_start", default=1., type=float)
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_exploration", default=1000000, type=float)
    ###########  DRQN  ###########
    parser.add_argument("--drqn_skill", default='norm', type = str, help="norm, doom")
    ################  Policy Based  ################
    parser.add_argument("--num_loss", default=1, type=int)
    parser.add_argument("--num_cpu", default=8, type=int)
    parser.add_argument("--all_cpu", type=str)
    ###############  Common Arguments  ###############
    parser.add_argument("--report_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--report_file_name", type = str)
    args = parser.parse_args()
    ##############################################
    if args.all_cpu == "True": args.num_cpu = multiprocessing.cpu_count()
    args.report_file_name = args.game + "_" + args.model + ".txt"
    args.report_path = "./report/"
    args.model_path = "./model/"+args.game+"/"
    make_path(args.report_path)
    make_path(args.model_path)

    if args.model == 'dqn': dqn.train(args)
    if args.model == 'drqn':
        if args.drqn_skill != 'mine' : drqn.train(args)
    if args.model == 'a3c': a3c.train(args)

if __name__ == "__main__":
    main()