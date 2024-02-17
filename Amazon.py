from bot_studio import *
amazon = bot_studio.amazon()
amazon.login(password='place password here', email='place email here')
amazon.buy(product_url='product_url')
amazon.select_payment_method(payment_method='Punjab National Bank Debit Card')
amazon.fill_cvv(cvv='cvv')
amazon.place_order()
