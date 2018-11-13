import argparse
import multiprocessing
import dqn, drqn, a3c
from utils.common_utils import make_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='a3c', type = str, help="dqn, drqn, a3c")
    ##############################################
    parser.add_argument("--game", default='BreakoutDeterministic', type = str)
    ################  Visual Attention  ################
    parser.add_argument("--tis", default='False', type = str)
    ################  Value Based  ################
    parser.add_argument("--double", default='False', type = str)
    parser.add_argument("--dueling", default='False', type = str)
    ###########  DRQN  ###########
    parser.add_argument("--drqn_skill", default='norm', type = str, help="norm, doom")
    ################  Policy Based  ################
    parser.add_argument("--num_cpu", default=8, type=int)
    parser.add_argument("--all_cpu", type=str)
    ###############  Common Arguments  ###############
    parser.add_argument("--train_time", default=24, type=int)
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
    if args.model == 'drqn': drqn.train(args)
    if args.model == 'a3c': a3c.train(args)


if __name__ == "__main__":
    main()