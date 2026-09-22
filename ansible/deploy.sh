#!/bin/bash

PLAYBOOK="$(dirname "$0")/ollama_agent_playbook.yml"
ANSIBLE_ENV="ANSIBLE_LOCALHOST_WARNING=false ANSIBLE_HOST_KEY_CHECKING=False"

PS3='Please enter your choice: '
OPTIONS=("Deploy the agent" "Dry run (check mode)" "Show the API key" "Quit")
select opt in "${OPTIONS[@]}"
do
    case $opt in
        "Deploy the agent")
            env $ANSIBLE_ENV ansible-playbook "$PLAYBOOK" --ask-pass --ask-become
            exit 0
        ;;
        "Dry run (check mode)")
            env $ANSIBLE_ENV ansible-playbook "$PLAYBOOK" --ask-pass --ask-become --check --diff
            exit 0
        ;;
        "Show the API key")
            env $ANSIBLE_ENV ansible-playbook "$PLAYBOOK" --ask-pass --ask-become --tags key
            exit 0
        ;;
        "Quit")
            echo "Script ended"
            break
        ;;
        *)
            echo "$REPLY is not a valid parameter"
            exit 1
        ;;
    esac
done
