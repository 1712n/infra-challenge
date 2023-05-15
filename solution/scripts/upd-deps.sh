#/bin/bash

repo_dir=`git rev-parse --show-toplevel`
echo -e "\033[1;34mStep 1: Activating venv ... \033[0m"
cd $repo_dir && source ./venv/Scripts/activate
echo -e "\033[1;34mStep 2: Installin pip-tools ...\033[0m"
pip install --upgrade pip setuptools wheel
pip install pip-tools
echo -e "\033[1;34mStep 3: Removing old deps ...\033[0m"
if compgen -G "$repo_dir/solution/requirements/requirements*.txt" > /dev/null; then
	rm $repo_dir/solution/requirements/requirements*.txt
fi
echo -e "\033[1;34mStep 4: Creating requirements.txt ...\033[0m"
pip-compile ./solution/requirements/requirements.in
echo -e "\033[1;34mStep 5: Creating requirements-dev.txt ...\033[0m"
pip-compile ./solution/requirements/requirements-dev.in