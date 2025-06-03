# NinjaFocus

Start Your Bot Manually
Now that the files are in place, you can start the systemd service.

From the SSH terminal:

Bash

    sudo systemctl daemon-reload # Reload systemd manager configuration
    sudo systemctl start trading_bot.service # Start the bot service

Check the bot's status and logs:
To see if your bot is running and view its logs:

Bash

    sudo systemctl status trading_bot.service
    sudo tail -f /var/log/trading_bot.log

The tail -f command will show you the real-time logs of your bot as they are written to the file. You should see your logger.info messages appearing here. Press Ctrl+C to exit tail -f.

If you see errors, check the error log:

Bash

    sudo tail -f /var/log/trading_bot_error.log


## Option 2

Activate the virtual environment:

Bash

source venv/bin/activate
After running this, your terminal prompt should change to something like (venv) mgtojar@solana-trading-bot-vm:~/app$, indicating that the virtual environment is active.

3. Install Your Bot's Dependencies within the Virtual Environment:

Now that the virtual environment is active, pip3 will refer to the pip within your virtual environment, and it will no longer give you the "externally-managed-environment" error.

Bash

    pip install -r requirements.txt
Notice it's pip here, not pip3, because the virtual environment's pip is now in your PATH.

4. Update Your systemd Service to Use the Virtual Environment:

Since your bot will now run within this virtual environment, you need to tell the systemd service to use the Python interpreter from inside the venv directory.

Edit the service file:

Bash

    sudo nano /etc/systemd/system/trading_bot.service
(Nano is a simple text editor. Use arrow keys to navigate. Ctrl+X to exit, Y to save, Enter to confirm filename.)

Find the `ExecStart` line and modify it:

Change this:

    ExecStart=/usr/bin/python3 /app/bot.py
To this:

    ExecStart=/app/venv/bin/python /app/bot.py
This tells systemd to use the Python interpreter specifically from your virtual environment.

Optional (and recommended for better logging): You can also add the WorkingDirectory if it's not already there (though your original setup script likely included it). This helps systemd know where to find relative paths if your bot uses any.

---

Store this in: /etc/systemd/system/trading_bot.service

[Unit]
Description=Python Trading Bot
After=network.target

[Service]
User=mgtojar
Group=mgtojar
WorkingDirectory=/home/mgtojar/app
ExecStart=/home/mgtojar/app/start_bot.sh
Restart=always
StandardOutput=append:/var/log/trading_bot.log
StandardError=append:/var/log/trading_bot.log

[Install]
WantedBy=multi-user.target

---

Save and Exit Nano: Press `Ctrl+X`, then `Y`, then `Enter`.

5. Reload systemd and Restart Your Bot:

Bash

    sudo systemctl daemon-reload
    sudo systemctl restart trading_bot.service
6. Verify:

Bash

    sudo systemctl status trading_bot.service
    sudo tail -f /var/log/trading_bot.log

Your bot should now start successfully and send messages. Using a virtual environment is the cleanest and most robust way to manage Python dependencies for applications like yours on Linux systems.

7. To kill the bot:

Stop the systemd service:

Bash

    sudo systemctl stop trading_bot.service

---

to copy paste:

[Unit]
Description=Python Trading Bot
After=network.target

[Service]
User=mgtojar
Group=mgtojar
WorkingDirectory=/home/mgtojar/app
ExecStart=/home/mgtojar/app/start_bot.sh
Restart=always
StandardOutput=append:/var/log/trading_bot.log
StandardError=append:/var/log/trading_bot.log

[Install]
WantedBy=multi-user.target

---

sudo systemctl daemon-reload
sudo systemctl restart trading_bot.service
sudo systemctl status trading_bot.service
sudo tail -f /var/log/trading_bot.log