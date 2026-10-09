---
doc_id: user-management-sso
title: Users, Roles and Single Sign-On
effective_date: 2026-01-20
status: current
---
# Users, Roles and Single Sign-On

## Roles

| Role | Can do |
|---|---|
| Owner | Everything, including billing and deleting the workspace |
| Admin | Manage members, integrations and settings, but not billing |
| Member | Create and edit projects and tasks |
| Guest | View shared projects and comment; cannot edit |

## Single sign-on (SSO)

SAML SSO works with Okta, Microsoft Entra ID and Google Workspace. SSO is included in the Business and Enterprise plans and is available as a paid add-on on the Team plan. Once SSO is enforced, users must sign in through the identity provider; owners keep a backup password login.

## Automatic user provisioning (SCIM)

SCIM provisioning creates and removes users automatically from your identity provider. It is available on the Enterprise plan only.

## Two-factor authentication

Every user can turn on two-factor authentication. On the Team, Business and Enterprise plans, admins can make it mandatory for the whole workspace.

## Removing users

When a user is removed, their tasks stay in the workspace and are reassigned to the person who removed them. Billing for removed users is covered in the Refund Policy.
