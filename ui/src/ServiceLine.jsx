import React from 'react'
import Grid from '@material-ui/core/Grid'
import Typography from '@material-ui/core/Typography'
import ExpansionPanel from '@material-ui/core/ExpansionPanel'
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary'
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails'
import ExpandMoreIcon from '@material-ui/icons/ExpandMore'

import Service from './Service'

function titleCase(s) {
  return s[0].toUpperCase() + s.slice(1).toLowerCase()
}

function normalizeName(name) {
  const parts = name.split('-')
  return parts.map(titleCase).join(' ')
}

const ServiceLine = (props) => {
  const taskName = normalizeName(props.task)
  const implementations = props.implementations.map(
    (i) => <Service
      desc={i}
      key={i.name}
      selected={i.name === props.selected}
      onSelect={props.onSelect}
    />
  )

  return (
    <Grid item xs={12}>
      <ExpansionPanel>
        <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant='h5'>{taskName}</Typography>
        </ExpansionPanelSummary>
        <ExpansionPanelDetails>
          {implementations}
        </ExpansionPanelDetails>
      </ExpansionPanel>
    </Grid>
  )
}

export default ServiceLine