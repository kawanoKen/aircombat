#!/bin/bash -e
# Qiita記事: https://qiita.com/yama07/items/b905ceff0498e52b00cb
# GitHub: https://github.com/yama07/docker-ubuntu-lxde/blob/master/xrdp/docker-entrypoint.sh
# ライセンスはCC0とのこと(2024.11.18時点)
#
# 上記元ファイルからの修正点
# (1) gid,uidの検索方法の改善
# https://create-it-myself.com/know-how/set-host-userid-groupid-to-docker-container/


USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER=${USER:-${DEFAULT_USER}}
GROUP=${GROUP:-${USER}}
PASSWD=${PASSWD:-${DEFAULT_PASSWD}}

unset DEFAULT_USER DEFAULT_PASSWD

# Add group
echo "GROUP_ID: $GROUP_ID"
if getent group "$GROUP_ID" > /dev/null 2>&1; then
    echo "[$SHELL_NAME] GROUP_ID '$GROUP_ID' already exists."
else
    echo "[$SHELL_NAME] GROUP_ID '$GROUP_ID' does NOT exist. So execute [groupadd -g \$GROUP_ID \$GROUP]."
    groupadd -g $GROUP_ID $GROUP
fi

# Add user
echo "USER_ID: $USER_ID"
if getent passwd "$USER_ID" > /dev/null 2>&1; then
    EXISTING_USER=$(getent passwd "$USER_ID" | cut -d: -f1)
    echo "[$SHELL_NAME] USER_ID '$USER_ID' already exists as '$EXISTING_USER'."
else
    echo "[$SHELL_NAME] USER_ID '$USER_ID' does NOT exist. So execute [useradd -d \$\{HOME\} -m -s /bin/bash -u \$USER_ID -g \$GROUP_ID \$USER_NAME]."
    export HOME=/home/$USER
    useradd -d ${HOME} -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER
fi

# Revert permissions
sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd

if (( $# == 0 )); then
    # Set login user name
    USER=$(whoami)
    echo "USER: $USER"

    # Set login password
    echo "PASSWD: $PASSWD"
    echo ${USER}:${PASSWD} | sudo chpasswd

    [[ ! -e ${HOME}/.xsession ]] && \
        cp /etc/skel/.xsession ${HOME}/.xsession
    [[ ! -e /etc/xrdp/rsakeys.ini ]] && \
        sudo -u xrdp -g xrdp xrdp-keygen xrdp /etc/xrdp/rsakeys.ini > /dev/null 2>&1

    RUNTIME_DIR=/run/user/${USER_ID}
    [ -e $RUNTIME_DIR ] && sudo rm -rf $RUNTIME_DIR
    sudo install -o $USER_ID -g $GROUP_ID -m 0700 -d $RUNTIME_DIR

    set -- /usr/bin/supervisord -c /etc/supervisor/xrdp.conf
    if [[ $USER_ID != "0" ]]; then
        [[ ! -e /usr/local/bin/_alt-su ]] && \
            sudo install -g $GROUP_ID -m 4750 $(which gosu || which su-exec) /usr/local/bin/_alt-su
        set -- /usr/local/bin/_alt-su root "$@"
    fi
fi
unset PASSWD

echo "#############################"
exec "$@"
