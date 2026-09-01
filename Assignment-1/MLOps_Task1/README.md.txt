# MLOps Task 1 – Olist Data Ingestion

## المشروع
تحميل بيانات Olist من Kaggle وتخزينها في قاعدة بيانات PostgreSQL.

## الجداول المنشأة (9 جداول)
- customers
- orders
- order_items
- products
- sellers
- order_payments
- order_reviews
- geolocation
- product_category_translation

## أعداد السجلات
- customers: 99,441
- orders: 99,441
- order_items: 112,650
- products: 32,951
- sellers: 3,095
- order_payments: 103,886
- order_reviews: 99,224
- geolocation: 1,000,163
- product_category_translation: 71

## العلاقات بين الجداول
- orders ↔ customers (customer_id)
- orders ↔ order_items (order_id)
- order_items ↔ products (product_id)
- order_items ↔ sellers (seller_id)

## المشكلة النهائية
التنبؤ إذا كان الطلب سيتأخر أم لا، بمقارنة تاريخ التسليم الفعلي مع التاريخ المتوقع.