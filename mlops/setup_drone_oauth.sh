#!/bin/bash
# setup_drone_oauth.sh: Automatically set up Gitea OAuth for Drone CI

set -e

# Variables
GITEA_URL="http://localhost:3000"
DRONE_URL="http://localhost:3001"
OAUTH_APP_NAME="Drone CI"
REDIRECT_URI="$DRONE_URL/login"

# Extract Gitea credentials from secret
YAML_DIR="k8s_yamls/gitea"
GITEA_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
GITEA_PASS=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')

echo "=========================================="
echo "Automated Gitea OAuth Setup for Drone CI"
echo "=========================================="
echo ""

# Wait for Gitea to be available
echo "Waiting for Gitea web server to be available..."
for i in {1..30}; do
    if curl -s --head --fail "$GITEA_URL" > /dev/null; then
        echo "Gitea web server is up."
        break
    else
        echo "Gitea web server not ready yet, retrying ($i)..."
        sleep 2
    fi
done

# Check if OAuth app already exists
echo "Checking for existing OAuth application..."

# First, try to get existing applications
EXISTING_APPS=$(curl -s -u "$GITEA_USER:$GITEA_PASS" "$GITEA_URL/api/v1/user/applications/oauth2" 2>/dev/null || echo "[]")

# Check if our app exists
if echo "$EXISTING_APPS" | grep -q "\"name\":\"$OAUTH_APP_NAME\""; then
    echo "OAuth application '$OAUTH_APP_NAME' already exists."
    echo "Extracting client ID and secret from existing application..."
    
    # Try to extract client ID and secret from existing apps
    CLIENT_ID=$(echo "$EXISTING_APPS" | grep -o '"client_id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
    CLIENT_SECRET=$(echo "$EXISTING_APPS" | grep -o '"client_secret":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
    
    if [ -n "$CLIENT_ID" ] && [ -n "$CLIENT_SECRET" ]; then
        echo "Found existing credentials."
    else
        echo "Could not extract credentials from existing application."
        echo "Please delete the existing application and rerun this script."
        exit 1
    fi
else
    echo "Creating OAuth application '$OAUTH_APP_NAME'..."
    RESPONSE=$(curl -s -X POST "$GITEA_URL/api/v1/user/applications/oauth2" \
        -u "$GITEA_USER:$GITEA_PASS" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$OAUTH_APP_NAME\",\"redirect_uris\":[\"$REDIRECT_URI\"]}" 2>/dev/null || echo "{}")
    
    # Check if response contains error
    if echo "$RESPONSE" | grep -q '"message"'; then
        echo "⚠️ API Error: $RESPONSE"
        echo "Please create the OAuth application manually."
        exit 1
    fi
    
    # Extract client_id and client_secret from response
    CLIENT_ID=$(echo "$RESPONSE" | grep -o '"client_id":"[^"]*"' | cut -d'"' -f4 || echo "")
    CLIENT_SECRET=$(echo "$RESPONSE" | grep -o '"client_secret":"[^"]*"' | cut -d'"' -f4 || echo "")
    
    if [ -n "$CLIENT_ID" ] && [ -n "$CLIENT_SECRET" ]; then
        echo "Successfully created OAuth application!"
        echo "Client ID: $CLIENT_ID"
        echo "Client Secret: $CLIENT_SECRET"
        
        # Update Drone secret YAML
        echo "Updating Drone secret YAML..."
        DRONE_SECRET_FILE="k8s_yamls/drone/02-secret.yaml"
        
        # Create backup
        cp "$DRONE_SECRET_FILE" "${DRONE_SECRET_FILE}.backup"
        
        # Update client ID and secret
        sed -i "s/drone-gitea-client-id: \".*\"/drone-gitea-client-id: \"$CLIENT_ID\"/" "$DRONE_SECRET_FILE"
        sed -i "s/drone-gitea-client-secret: \".*\"/drone-gitea-client-secret: \"$CLIENT_SECRET\"/" "$DRONE_SECRET_FILE"
        
        echo "Updated $DRONE_SECRET_FILE"
        
        # Apply updated secret
        echo "Applying updated secret to Kubernetes..."
        kubectl apply -f "$DRONE_SECRET_FILE"
        
        # Restart Drone deployment
        echo "Restarting Drone deployment..."
        kubectl rollout restart deployment/drone -n drone
        
        echo ""
        echo "✅ OAuth setup completed successfully!"
        echo "Drone CI is now configured with Gitea OAuth."
        echo "Access Drone at: $DRONE_URL"
    else
        echo "⚠️ Failed to extract client ID and secret from response."
        echo "Response: $RESPONSE"
        echo ""
        echo "Please create the OAuth application manually:"
        echo "1. Go to: $GITEA_URL/user/settings/applications"
        echo "2. Login with: $GITEA_USER / $GITEA_PASS"
        echo "3. Create application:"
        echo "   - Name: $OAUTH_APP_NAME"
        echo "   - Redirect URI: $REDIRECT_URI"
        echo "4. Update k8s_yamls/drone/02-secret.yaml with client ID/secret"
        echo "5. Run: kubectl apply -f k8s_yamls/drone/02-secret.yaml"
        echo "6. Run: kubectl rollout restart deployment/drone -n drone"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Setup Summary:"
echo "=========================================="
echo "Gitea URL: $GITEA_URL"
echo "Drone URL: $DRONE_URL"
echo "OAuth App: $OAUTH_APP_NAME"
echo "Redirect URI: $REDIRECT_URI"
echo ""
echo "To verify OAuth setup:"
echo "1. Go to: $GITEA_URL/user/settings/applications"
echo "2. Login with Gitea credentials"
echo "3. Check that '$OAUTH_APP_NAME' application exists"
echo ""
echo "To access Drone CI:"
echo "1. Go to: $DRONE_URL"
echo "2. Click 'Login with Gitea'"
echo "3. Authorize the application"
echo "=========================================="