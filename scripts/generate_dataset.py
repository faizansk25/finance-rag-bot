"""
Synthetic Invoice Dataset Generator

Generates realistic finance documents:
- Invoices with proper structure (header, line items, terms)
- Purchase orders
- Payment receipts

Each document has realistic metadata for domain-aware chunking evaluation.
"""

import random
import json
from datetime import datetime, timedelta
from typing import List


VENDORS = [
    {"name": "Acme Corp", "gl_prefix": "5000"},
    {"name": "GlobalTech Solutions", "gl_prefix": "5100"},
    {"name": "Premier Office Supplies", "gl_prefix": "5200"},
    {"name": "CloudHost Inc", "gl_prefix": "5300"},
    {"name": "DataFlow Systems", "gl_prefix": "5400"},
    {"name": "SecureNet Labs", "gl_prefix": "5500"},
    {"name": "GreenPrint Services", "gl_prefix": "5600"},
    {"name": "FastShip Logistics", "gl_prefix": "5700"},
    {"name": "BrightMind AI", "gl_prefix": "5800"},
    {"name": "CircuitBoard Manufacturing", "gl_prefix": "5900"},
]

LINE_ITEMS = [
    {"desc": "Cloud Hosting (Monthly)", "qty": 1, "unit_price": 299.99, "gl": "5300"},
    {"desc": "Software License - Annual", "qty": 1, "unit_price": 1499.00, "gl": "5100"},
    {"desc": "Office Paper (5000 sheets)", "qty": 10, "unit_price": 24.99, "gl": "5200"},
    {"desc": "API Calls (100K batch)", "qty": 5, "unit_price": 49.99, "gl": "5400"},
    {"desc": "Security Audit Service", "qty": 1, "unit_price": 3500.00, "gl": "5500"},
    {"desc": "Printer Toner Cartridge", "qty": 4, "unit_price": 89.99, "gl": "5600"},
    {"desc": "Shipping Container (Large)", "qty": 2, "unit_price": 199.99, "gl": "5700"},
    {"desc": "ML Model Training (GPU Hours)", "qty": 20, "unit_price": 15.00, "gl": "5800"},
    {"desc": "Circuit Board Assembly", "qty": 50, "unit_price": 12.50, "gl": "5900"},
    {"desc": "Consulting Services (10 hrs)", "qty": 10, "unit_price": 250.00, "gl": "5000"},
    {"desc": "Data Backup Storage (TB)", "qty": 5, "unit_price": 9.99, "gl": "5300"},
    {"desc": "Network Cable (Cat6, 100ft)", "qty": 20, "unit_price": 34.99, "gl": "5900"},
]

PAYMENT_TERMS = [
    "Net 30",
    "Net 15",
    "Net 45",
    "Due on Receipt",
    "2/10 Net 30",
]

STATUSES = ["paid", "pending", "overdue", "approved", "draft"]


def random_date(start_year: int = 2024, end_year: int = 2026) -> str:
    """Generate a random date string."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    random_days = random.randint(0, delta)
    date = start + timedelta(days=random_days)
    return date.strftime("%m/%d/%Y")


def generate_invoice(invoice_id: int) -> str:
    """Generate a single realistic invoice document."""
    vendor = random.choice(VENDORS)
    status = random.choice(STATUSES)
    invoice_date = random_date()
    payment_terms = random.choice(PAYMENT_TERMS)

    # Calculate due date based on terms
    inv_date = datetime.strptime(invoice_date, "%m/%d/%Y")
    if "Net 15" in payment_terms:
        due_date = inv_date + timedelta(days=15)
    elif "Net 30" in payment_terms:
        due_date = inv_date + timedelta(days=30)
    elif "Net 45" in payment_terms:
        due_date = inv_date + timedelta(days=45)
    else:
        due_date = inv_date
    due_date_str = due_date.strftime("%m/%d/%Y")

    # Select 2-4 random line items
    num_items = random.randint(2, 4)
    items = random.sample(LINE_ITEMS, num_items)

    # Calculate totals
    subtotal = sum(item["qty"] * item["unit_price"] for item in items)
    tax_rate = random.choice([0.0, 0.05, 0.08, 0.10])
    tax = subtotal * tax_rate
    total = subtotal + tax

    # Build invoice text
    invoice_num = f"INV-{invoice_id:04d}"

    lines = []
    lines.append(f"Invoice")
    lines.append(f"Invoice Number: {invoice_num}")
    lines.append(f"Date: {invoice_date}")
    lines.append(f"Due Date: {due_date_str}")
    lines.append(f"Status: {status}")
    lines.append(f"")
    lines.append(f"From:")
    lines.append(f"Vendor: {vendor['name']}")
    lines.append(f"GL Account: {vendor['gl_prefix']}")
    lines.append(f"")
    lines.append(f"To:")
    lines.append(f"Acme Industries Inc.")
    lines.append(f"123 Business Ave")
    lines.append(f"San Francisco, CA 94105")
    lines.append(f"")
    lines.append(f"Item No    Description                          Qty    Unit Price    Amount")
    lines.append(f"--------   -----------------------------------  -----  ----------   ----------")

    for i, item in enumerate(items, 1):
        amount = item["qty"] * item["unit_price"]
        lines.append(
            f"{i:<9}  {item['desc']:<36} {item['qty']:<6} ${item['unit_price']:>9.2f}   ${amount:>9.2f}"
        )

    lines.append(f"")
    lines.append(f"{'':>55} Subtotal:    ${subtotal:>9.2f}")
    lines.append(f"{'':>55} Tax ({tax_rate:.0%}):  ${tax:>9.2f}")
    lines.append(f"{'':>55} Total:       ${total:>9.2f}")
    lines.append(f"")
    lines.append(f"Payment Terms: {payment_terms}")
    lines.append(f"")
    lines.append(f"Bank Details:")
    lines.append(f"Bank: First National Bank")
    lines.append(f"Account: ****-****-****-{random.randint(1000, 9999)}")
    lines.append(f"Routing: {random.randint(100000000, 999999999)}")
    lines.append(f"")
    lines.append(f"Notes: Please reference invoice number {invoice_num} on your payment.")

    return "\n".join(lines)


def generate_sample_invoices(count: int = 20) -> List[str]:
    """Generate multiple sample invoices."""
    invoices = []
    for i in range(1, count + 1):
        invoices.append(generate_invoice(i))
    return invoices


def generate_dataset(output_dir: str = "data", count: int = 50):
    """Generate a full dataset and save to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    invoices = generate_sample_invoices(count)

    # Save individual invoices
    for i, inv in enumerate(invoices, 1):
        filepath = os.path.join(output_dir, f"invoice_{i:04d}.txt")
        with open(filepath, 'w') as f:
            f.write(inv)

    # Save combined dataset
    combined_path = os.path.join(output_dir, "invoices.json")
    dataset = []
    for i, inv in enumerate(invoices, 1):
        dataset.append({
            "id": f"INV-{i:04d}",
            "text": inv,
            "doc_type": "invoice",
            "source": f"invoice_{i:04d}.txt",
        })

    with open(combined_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {count} invoices in {output_dir}/")
    print(f"  - Individual files: {output_dir}/invoice_*.txt")
    print(f"  - Combined dataset: {combined_path}")

    return invoices


if __name__ == "__main__":
    generate_dataset(count=50)
