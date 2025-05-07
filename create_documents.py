import os
import sys

COMPANY_FAQ = """# TechCorp Frequently Asked Questions

## About TechCorp
TechCorp was founded in 2010 by Jane Smith and Michael Chen. Our mission is to create innovative software solutions that make businesses more efficient and productive. We currently have offices in San Francisco, New York, and London, with over 500 employees worldwide.

## Products and Services
TechCorp offers a range of enterprise software solutions:
- TechFlow: Our flagship project management software
- DataSense: Business intelligence and analytics platform
- CloudConnect: Secure cloud storage and file sharing
- SecureGuard: Cybersecurity protection suite

## Pricing
Our products are available through subscription models:
- Basic: $49/month per user
- Professional: $99/month per user
- Enterprise: Custom pricing, contact sales

## Technical Support
Technical support is available 24/7 through:
- Email: support@techcorp.com
- Phone: 1-800-TECH-HELP
- Live Chat: Available on our website

## Return Policy
We offer a 30-day money-back guarantee on all subscription plans. If you're not satisfied with our products, you can cancel within 30 days for a full refund.
"""

PRODUCT_SPECS = """# TechFlow Product Specifications

## Overview
TechFlow is a comprehensive project management solution designed for modern teams. It combines task management, resource allocation, time tracking, and collaboration tools in one intuitive platform.

## Key Features
- Interactive Gantt charts and Kanban boards
- Real-time collaboration with team members
- Resource management and capacity planning
- Time tracking and reporting
- Customizable workflows and automation
- Integration with 100+ popular tools
- Mobile apps for iOS and Android

## Technical Specifications
- Cloud-based SaaS application
- 99.9% uptime guaranteed
- AES-256 encryption for all data
- GDPR and CCPA compliant
- Regular automatic backups
- API access for custom integrations

## System Requirements
- Web application: Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile apps: iOS 13+ or Android 8+
- Internet connection: Minimum 5 Mbps

## Deployment Options
- Cloud-hosted (standard)
- On-premise deployment (enterprise plans only)
- Hybrid solutions available
"""

EMPLOYEE_HANDBOOK = """# TechCorp Employee Handbook

## Company Values
At TechCorp, we believe in:
- Innovation: Constantly seeking new and better solutions
- Integrity: Being honest and transparent in all we do
- Collaboration: Working together to achieve shared goals
- Customer Focus: Putting our users' needs first
- Diversity: Embracing different perspectives and backgrounds

## Work Schedule
- Standard work hours are 9:00 AM to 5:00 PM local time
- Flexible work arrangements available upon manager approval
- Remote work options available for most positions
- Core hours (when all employees should be available): 11:00 AM to 3:00 PM

## Benefits
- Comprehensive health insurance (medical, dental, vision)
- 401(k) retirement plan with company matching
- 20 days of paid time off per year
- 10 paid holidays annually
- Parental leave: 16 weeks paid
- Professional development budget: $2,000 annually
- Wellness program with gym reimbursement

## Expense Policy
Employees may be reimbursed for reasonable business expenses, including:
- Business travel (flights, accommodations, meals)
- Client meetings and entertainment
- Professional development and education
- Required tools and equipment

All expenses must be approved by a manager and submitted with receipts within 30 days.
"""

def main():
    if not os.path.exists('documents'):
        os.makedirs('documents')
        print("Created 'documents' directory.")
    try:
        with open('documents/company_faq.txt', 'w') as f:
            f.write(COMPANY_FAQ)
        
        with open('documents/product_specs.txt', 'w') as f:
            f.write(PRODUCT_SPECS)
        
        with open('documents/employee_handbook.txt', 'w') as f:
            f.write(EMPLOYEE_HANDBOOK)
        
        print("Successfully created sample documents in the 'documents' directory:")
        print("1. company_faq.txt")
        print("2. product_specs.txt")
        print("3. employee_handbook.txt")
        
    except Exception as e:
        print(f"Error creating documents: {str(e)}")
        sys.exit(1)
    
    print("\nYou can now run the RAG assistant with: python rag_agent.py")

if __name__ == "__main__":
    main()