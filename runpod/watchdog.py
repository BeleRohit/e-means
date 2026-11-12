"""Local deadline guard for a Pod created by this explicitly approved session.

No provisioning. Keep this controller host awake and online. At deadline only
the Pod ID from the session's own saved create response is terminated. A
network volume is retained for artifact recovery; normal cleanup must remove
it after verified local backup. This is not a provider-enforced dollar cap.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import time


def terminate(pod_id,run=subprocess.run):
    if not isinstance(pod_id,str) or not pod_id.isalnum():raise ValueError('invalid saved Pod ID')
    def call(*args):return run(['runpodctl','pod',*args],capture_output=True,text=True,timeout=45)
    before=call('get',pod_id)
    if before.returncode:
        if '"not_found"' in before.stderr:return True
        return False
    if json.loads(before.stdout).get('id')!=pod_id:raise RuntimeError('readback identity mismatch')
    deleted=call('delete',pod_id)
    if deleted.returncode and '"not_found"' not in deleted.stderr:return False
    after=call('get',pod_id)
    return after.returncode!=0 and '"not_found"' in after.stderr


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();budget=json.loads((args.session/'budget.json').read_text())
    if budget.get('explicitly_approved') is not True:raise ValueError('record explicit user approval before starting guard')
    deadline=float(budget['deadline_unix'])
    if not math.isfinite(deadline) or deadline<=0:raise ValueError('finite positive deadline required')
    if args.dry_run:
        print('No cloud calls: guard would terminate only the saved session Pod after',deadline);return
    with (args.session/'watchdog.log').open('a',buffering=1) as log:
        log.write('Started '+datetime.now(timezone.utc).isoformat()+'\n')
        while time.time()<deadline:
            if (args.session/'cleanup-complete').exists():log.write('Normal cleanup complete\n');return
            time.sleep(min(5,max(0,deadline-time.time())))
        create=args.session/'pod.json'
        if not create.exists():log.write('No saved Pod creation response; nothing to terminate\n');return
        pod=json.loads(create.read_text())['id']
        for attempt in range(30):
            try:
                if terminate(pod):log.write('Pod termination verified: '+pod+'\n');return
            except (subprocess.TimeoutExpired,OSError,ValueError) as error:
                log.write('Retry '+str(attempt)+' '+type(error).__name__+'\n')
            time.sleep(10)
        raise RuntimeError('Pod termination not verified; controller or API unavailable')


if __name__=='__main__':main()
