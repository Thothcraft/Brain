"""ssh helper:
  python scripts/ssh_node.py <host> "<remote command>"
  python scripts/ssh_node.py <host> --script <local_file>   (bash <file>)
"""
import sys
import paramiko

host = sys.argv[1]
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(host, username="gad", password="password", timeout=8,
            look_for_keys=False, allow_agent=False)

if sys.argv[2] == "--script":
    local = sys.argv[3].replace("\\", "/")
    remote = "/tmp/_ssh_" + local.rsplit("/", 1)[-1]
    sftp = cli.open_sftp()
    with open(local, "rb") as fh, sftp.open(remote, "wb") as rh:
        rh.write(fh.read())
    sftp.close()
    cmd = f"bash {remote}"
else:
    cmd = sys.argv[2]

_, out, err = cli.exec_command(cmd, timeout=60)
sys.stdout.write(out.read().decode(errors="replace"))
sys.stderr.write(err.read().decode(errors="replace"))
cli.close()
