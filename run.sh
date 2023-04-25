#! /usr/bin/env bash

export PROJECT_NAME=verti_wheelers  # add your project folder to python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
export COMET_LOGGING_CONSOLE=info

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation."
   echo 
   echo "options:"
   echo "train                      Starts training."
   echo "parse                      Parse bag files."
   echo "run file_name              Runs file_name.py file."
   echo
}

run () {
  case $1 in
    train)
      python $PROJECT_NAME/main.py --conf $PROJECT_NAME/conf/config
      ;;
    parse)
      python $PROJECT_NAME/utils/parser.py --conf $PROJECT_NAME/conf/parser_config
      ;;
    run)
      python $2
      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2

# echo "Done."
