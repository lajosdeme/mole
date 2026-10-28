package llm

import (
	"context"
	"strings"
)

// ModelRole names a job a model does in a session.
type ModelRole string

const (
	// RoleStrong plans leads and writes the report.
	RoleStrong ModelRole = "strong"
	// RoleCheap mines chunks for claims.
	RoleCheap ModelRole = "cheap"
	// RoleVerifier judges claim pairs and grounding.
	RoleVerifier ModelRole = "verifier"
)

// Roles lists every role in display order.
var Roles = []ModelRole{RoleStrong, RoleCheap, RoleVerifier}

// matchOrder is the precedence for routing by model name.
var matchOrder = []ModelRole{RoleVerifier, RoleCheap, RoleStrong}

type router struct {
	roles  map[ModelRole]Provider
	models map[ModelRole]string
}

// NewRouter returns a Provider that sends each request to its role's provider.
func NewRouter(roles map[ModelRole]Provider, models map[ModelRole]string) Provider {
	return &router{roles: roles, models: models}
}

// Complete routes by explicit model first, verifier before cheap before strong, then by tier.
func (r *router) Complete(ctx context.Context, req Request) (*Response, error) {
	role := r.roleFor(req)
	if req.Model == "" {
		req.Model = r.models[role]
	}
	return r.roles[role].Complete(ctx, req)
}

func (r *router) roleFor(req Request) ModelRole {
	if req.Model != "" {
		for _, role := range matchOrder {
			if r.models[role] == req.Model && r.roles[role] != nil {
				return role
			}
		}
	}
	if req.Tier == TierCheap {
		return RoleCheap
	}
	return RoleStrong
}

// Name joins the distinct provider names with "+".
func (r *router) Name() string {
	var names []string
	for _, role := range Roles {
		p := r.roles[role]
		if p == nil {
			continue
		}
		if n := p.Name(); !containsString(names, n) {
			names = append(names, n)
		}
	}
	return strings.Join(names, "+")
}

// ModelFor returns the model of the role that serves the tier.
func (r *router) ModelFor(tier Tier) string {
	if tier == TierCheap {
		return r.models[RoleCheap]
	}
	return r.models[RoleStrong]
}

// CredentialSource is the roles' shared source, or CredentialConfig when they differ.
func (r *router) CredentialSource() CredentialSource {
	var src CredentialSource
	for _, p := range r.roles {
		s := SourceOf(p)
		if src != "" && s != src {
			return CredentialConfig
		}
		src = s
	}
	return src
}

func containsString(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}
