# Small hand-written seed dataset of scam vs legit messages/call-lines.
# This is intentionally small & illustrative — swap in a real labeled
# dataset (e.g. SMS Spam Collection, or your own transcripts) for
# production use. See README for where to plug that in.

SCAM_EXAMPLES = [
    "Congratulations! You have won a lottery of $1,000,000. Send your bank details to claim now.",
    "This is IRS. You owe back taxes and a warrant is issued for your arrest. Call us immediately or pay via gift cards.",
    "Your account will be suspended in 24 hours. Verify your login details here: bit.ly/verify-now",
    "Hi, this is Microsoft support, we detected a virus on your PC. Please install AnyDesk so we can fix it remotely.",
    "URGENT: Your package is held at customs. Pay a clearance fee of $50 via Western Union to release it.",
    "We noticed suspicious activity on your bank account. Please confirm your card number and CVV to secure it.",
    "You've been selected for a free iPhone! Click here to claim your prize before it expires today.",
    "This is your bank's security team. Do not tell anyone, but we need your OTP to reverse a fraudulent charge.",
    "Final notice: pay your outstanding loan today or legal action will be taken against you.",
    "Hello, I am calling from the government office. Your social security number has been suspended due to fraud, press 1 to speak to an officer.",
    "Your Amazon account has been locked. Verify your account within 1 hour to avoid permanent suspension.",
    "I'm a prince from a foreign country and need help transferring my inheritance, you'll get 20% for helping.",
    "Dear customer, update your KYC immediately or your SIM card will be deactivated tonight.",
    "This is the courier company, we need a small customs fee paid via UPI to deliver your parcel today.",
    "Your electricity bill is overdue, your connection will be cut in 30 minutes, pay now via this link.",
    "Hi grandma, it's me, I'm in jail and need bail money urgently, please don't tell mom and dad, send cash via MoneyGram.",
    "Tech support here — we saw errors on your computer, download this remote tool so we can fix it right away.",
    "Your card has been charged $499 for a subscription, if this wasn't you call this number immediately to cancel.",
    "You qualify for a government grant of $9500, just pay a small processing fee to receive the funds.",
    "This is your bank calling. For your safety, do not hang up and do not discuss this call with anyone, we need to verify your PIN.",
]

LEGIT_EXAMPLES = [
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "Your order #4521 has shipped and will arrive on Thursday. Track it in the app.",
    "Reminder: your dentist appointment is scheduled for June 10th at 3pm.",
    "Thanks for the update, I'll review the document tonight and get back to you.",
    "This is a reminder your electricity bill of $85 is due on the 15th. You can pay it through the usual portal.",
    "Hi, just checking in — how did the interview go?",
    "Your Netflix subscription renews on the 20th for $15.99. No action needed.",
    "Can you send me the notes from today's meeting?",
    "Your flight AI 202 is confirmed for July 12, departing at 6:45 AM.",
    "Mom, I landed safely, will call you once I reach the hotel.",
    "The package you ordered was delivered to your front porch at 2:14 PM today.",
    "Team, standup is moved to 10am tomorrow instead of 9:30.",
    "Your OTP for logging into your account is 482913. Do not share this with anyone.",
    "Happy birthday! Hope you have an amazing day.",
    "Your appointment with Dr. Lee has been confirmed for next Monday.",
    "Hey, I left my charger at your place, can I grab it this weekend?",
    "Invoice #2291 has been paid, thank you for your business.",
    "Let's catch up this weekend, it's been a while!",
    "Your library book is due back in 3 days.",
    "The meeting notes have been shared in the drive folder, take a look when free.",
]
