import can
import cantools

def load_dbc(dbc_file_path):
    return cantools.database.load_file(dbc_file_path)

def encode_scaled_scores_average(db, message_id, scaled_scores_average):
    message = db.get_message_by_name('MY_MESSAGE')
    data = message.encode({'scaled_scores_average': scaled_scores_average})
    return can.Message(arbitration_id=message_id, data=data, is_extended_id=False)

def send_can_message(bus, message):
    with can.interface.Bus(bus, bustype='socketcan') as bus:
        bus.send(message)
        print(f"Message sent on {bus.channel_info}")

def main():
    dbc_file_path = './score.dbc'
    db = load_dbc(dbc_file_path)

    scaled_scores_average_value = 75.0

    message = encode_scaled_scores_average(db, 0x500, scaled_scores_average_value)

    send_can_message('can0', message)

if __name__ == "__main__":
    main()
