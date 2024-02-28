label_idx_names = {}
merge_labels = 0
#merge_labels_dict = {2: 0, 1: 3}
max_size = 800
last_best_epoch = -1
epoch_num_wait_for_new_best = 500
save_epoch_after = 0.6
resume_load_optimizer = True
resume_load_epoch_num = True

def notify_by_email():
    import win32com.client as win32
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'toseedrina@gmail.com'
    mail.Subject = 'Training Finished'
    mail.Body = ''
    mail.HTMLBody = '<h2>Training Finished</h2>' #this field is optional
    mail.Send()
