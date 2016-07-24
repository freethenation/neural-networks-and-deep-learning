from fabric.operations import sudo, run

def setup():
    sudo('umount /mnt')
    sudo('chmod 0777 /mnt')
    run('scp freethenation@icyego.com:~/data_small.zip /mnt/')
    run('cd /mnt; git clone https://github.com/freethenation/neural-networks-and-deep-learning.git')
    