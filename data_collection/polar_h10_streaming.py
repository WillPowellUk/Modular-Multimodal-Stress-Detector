<<<<<<< HEAD
import os
import sys
import asyncio

# Add submodules to the system path
sys.path.append(r'submodules')

# Import the specific function or module from bleakheart
from bleakheart.examples import ecg_rr_acc_recorder as recorder # type: ignore

async def main(file_name):
    # Call the main function from ecg_queue, assuming it's an async function
    await recorder.main(file_name)

# Use asyncio.run() to run the main coroutine
if __name__ == "__main__":
    subject = 2
    file_name = f'data_collection/recordings/S{subject}'
    asyncio.run(main(file_name))
=======
import os
import sys
import asyncio

# Add submodules to the system path
sys.path.append(r'submodules')

# Import the specific function or module from bleakheart
from bleakheart.examples import ecg_rr_acc_recorder as recorder # type: ignore

async def main(file_name):
    # Call the main function from ecg_queue, assuming it's an async function
    await recorder.main(file_name)

# Use asyncio.run() to run the main coroutine
if __name__ == "__main__":
    subject = 8
    file_name = f'data_collection/recordings/S{subject}'
    asyncio.run(main(file_name))
>>>>>>> b9c557b039ff0564b53642fa18f7e0c49e63911d
